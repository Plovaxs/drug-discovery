#!/usr/bin/env bash
# Runs a GPU job chain only once the card is actually free AND has cooled down.
#
# Why this exists. This laptop has a 4GB card; two training jobs do not fit, so follow-on work has to
# wait rather than share. Waiting by hand means either babysitting a terminal for ten hours or coming
# back to an idle GPU. And the project's established practice is a cooling break between heavy runs, not
# back-to-back thermal load -- so "wait for free" alone is not the right condition either.
#
# Three guards, each from a mistake already made here:
#   1. pgrep uses the bracket trick ("train[.]py") so the pattern cannot match this script's own command
#      line. A plain `pkill -f` once killed the shell that issued it (exit 144).
#   2. Every step's exit code is checked EXPLICITLY. `set -e` did not abort a previous chain: all three
#      seeds ran, all three failed, and the chain reported success. A chain that cannot fail loudly is
#      worse than no chain, because it produces a confident "done" with no results behind it.
#   3. The thermal gate requires the temperature to stay under the threshold for a sustained window, not
#      merely touch it once. A single cool sample right after a job exits says nothing -- the card dumps
#      heat for a while after load ends.
#
# Usage:
#   scripts/queued_run.sh --wait-for "train_surrogate_arm[.]py" --log logs/queue.log -- <command> [args]
#   scripts/queued_run.sh --cool-to 55 --cool-for 600 --log logs/queue.log -- bash my_chain.sh
#
# Options:
#   --wait-for PATTERN   pgrep -f pattern that must have NO matches before starting (repeatable)
#   --cool-to C          start only below this GPU temperature in Celsius (default 55)
#   --cpu-cool-to C      ...and below this CPU package temperature (default 80; 0 disables)
#   --cool-for SECONDS   how long BOTH must stay below those, continuously (default 600)
#   --poll SECONDS       polling interval (default 60)
#   --max-wait SECONDS   give up waiting after this long (default 86400; 0 = forever)
#   --skip-cool          free-GPU check only, no thermal gate
#   --log FILE           tee all output here as well as stdout

set -uo pipefail

PATTERNS=()
COOL_TO=55
CPU_COOL_TO=80
COOL_FOR=600
POLL=60
MAX_WAIT=86400
SKIP_COOL=0
LOGFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-for) PATTERNS+=("$2"); shift 2 ;;
    --cool-to)  COOL_TO="$2"; shift 2 ;;
    --cpu-cool-to) CPU_COOL_TO="$2"; shift 2 ;;
    --cool-for) COOL_FOR="$2"; shift 2 ;;
    --poll)     POLL="$2"; shift 2 ;;
    --max-wait) MAX_WAIT="$2"; shift 2 ;;
    --skip-cool) SKIP_COOL=1; shift ;;
    --log)      LOGFILE="$2"; shift 2 ;;
    --)         shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given after --" >&2
  exit 2
fi

if [[ -n "$LOGFILE" ]]; then
  mkdir -p "$(dirname "$LOGFILE")"
  exec > >(tee -a "$LOGFILE") 2>&1
fi

log() { echo "[queued_run $(date '+%F %T')] $*"; }

gpu_temp() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
}

cpu_temp() {
  # Measured on this machine: with Arm B training and a load average of only 1.3 out of 20 threads, the
  # GPU sits at 72 C while x86_pkg_temp reads 87-94 C and TCPU/acpitz read 92 C. The CPU, not the GPU, is
  # this laptop's thermal bottleneck -- an i7-12700H whose Tjmax is 100 C, so 92 C is 8 C from throttling
  # on essentially one busy thread (the dataloader).
  #
  # Gating on GPU temperature alone therefore watched the wrong sensor: it would happily start more work
  # while the hot component was nearly maxed. The maximum across the CPU zones is used rather than any
  # single one, because they disagree by up to 7 C and the conservative reading is the safe one.
  local best="" t ty z
  for z in /sys/class/thermal/thermal_zone*/; do
    ty=$(cat "$z/type" 2>/dev/null) || continue
    case "$ty" in
      x86_pkg_temp|TCPU|acpitz|coretemp*|k10temp*) ;;
      *) continue ;;
    esac
    t=$(cat "$z/temp" 2>/dev/null) || continue
    [ -z "$t" ] && continue
    t=$(( t / 1000 ))
    # Reject implausible readings rather than letting a broken zone block the queue forever.
    [ "$t" -lt 10 ] || [ "$t" -gt 130 ] && continue
    if [ -z "$best" ] || [ "$t" -gt "$best" ]; then best=$t; fi
  done
  echo "$best"
}

busy() {
  # A match only counts if the matched process is really a python interpreter.
  #
  # This is not a refinement, it is a deadlock fix. `pgrep -f` matches against the WHOLE command line,
  # and this script's own command line contains the training script's path as an argument -- so
  # `--wait-for "train_egnn_heteroscedastic[.]py"` matched this very wrapper (and the subshell that
  # command substitution forks from it, which inherits the same argv). The bracket trick prevents pgrep
  # from matching the pgrep, but not from matching the caller. Left unfixed, the first chain step whose
  # own script name appeared in its own wait pattern would have waited on itself forever: no error, no
  # output, just a queue that never fires.
  #
  # /proc/<pid>/comm is the discriminator because it holds the EXECUTABLE name, not the argument vector:
  # a real training job reports "python", this wrapper and its subshells report "bash".
  local p pid comm
  for p in "${PATTERNS[@]:-}"; do
    [[ -z "$p" ]] && continue
    for pid in $(pgrep -f "$p" 2>/dev/null); do
      comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
      case "$comm" in
        python*|Python*) ;;
        *) continue ;;
      esac
      echo "$p (pid $pid, $comm)"
      return 0
    done
  done
  return 1
}

log "queued: $*"
if [[ ${#PATTERNS[@]} -gt 0 ]]; then
  log "waiting for these to clear: ${PATTERNS[*]}"
fi
[[ $SKIP_COOL -eq 0 ]] && log "thermal gate: GPU <${COOL_TO}C AND CPU <${CPU_COOL_TO}C, both sustained ${COOL_FOR}s"

START=$(date +%s)
COOL_SINCE=0

while true; do
  NOW=$(date +%s)
  ELAPSED=$(( NOW - START ))
  if [[ "$MAX_WAIT" -gt 0 && "$ELAPSED" -gt "$MAX_WAIT" ]]; then
    log "GAVE UP after ${ELAPSED}s without the start conditions being met; command NOT run"
    exit 75   # EX_TEMPFAIL: distinguishable from the command itself failing
  fi

  if BLOCKER=$(busy); then
    log "still busy: $BLOCKER (waited ${ELAPSED}s)"
    COOL_SINCE=0
    sleep "$POLL"
    continue
  fi

  if [[ $SKIP_COOL -eq 1 ]]; then
    log "GPU free; thermal gate skipped"
    break
  fi

  TEMP=$(gpu_temp)
  CTEMP=$(cpu_temp)
  if [[ -z "$TEMP" ]]; then
    # No reading is not permission to proceed: a job may be running on a card we cannot see.
    log "WARN cannot read GPU temperature; holding"
    COOL_SINCE=0
    sleep "$POLL"
    continue
  fi

  GPU_OK=0; CPU_OK=0
  [[ "$TEMP" -lt "$COOL_TO" ]] && GPU_OK=1
  if [[ "$CPU_COOL_TO" -eq 0 || -z "$CTEMP" ]]; then
    CPU_OK=1          # gate disabled, or no CPU sensor on this machine
  elif [[ "$CTEMP" -lt "$CPU_COOL_TO" ]]; then
    CPU_OK=1
  fi
  CDESC="CPU ${CTEMP:-?}C/${CPU_COOL_TO}C"
  [[ "$CPU_COOL_TO" -eq 0 ]] && CDESC="CPU gate off"

  if [[ "$GPU_OK" -eq 1 && "$CPU_OK" -eq 1 ]]; then
    if [[ "$COOL_SINCE" -eq 0 ]]; then
      COOL_SINCE=$NOW
      log "GPU ${TEMP}C < ${COOL_TO}C and ${CDESC} -- starting the ${COOL_FOR}s cool-down window"
    fi
    HELD=$(( NOW - COOL_SINCE ))
    if [[ "$HELD" -ge "$COOL_FOR" ]]; then
      log "both held cool for ${HELD}s (GPU ${TEMP}C, ${CDESC}); proceeding"
      break
    fi
    log "cool for ${HELD}/${COOL_FOR}s (GPU ${TEMP}C, ${CDESC})"
  else
    [[ "$COOL_SINCE" -ne 0 ]] && log "temperature rose again; cool-down window reset"
    COOL_SINCE=0
    WHY=""
    [[ "$GPU_OK" -eq 0 ]] && WHY="GPU ${TEMP}C >= ${COOL_TO}C"
    [[ "$CPU_OK" -eq 0 ]] && WHY="${WHY:+$WHY, }CPU ${CTEMP}C >= ${CPU_COOL_TO}C"
    log "waiting: $WHY"
  fi
  sleep "$POLL"
done

log "START: $*"
"$@"
rc=$?
if [[ $rc -ne 0 ]]; then
  log "FAILED with exit $rc: $*"
else
  log "COMPLETED: $*"
fi
exit $rc
