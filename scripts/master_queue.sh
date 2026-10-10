#!/usr/bin/env bash
# Serial work queue with enforced cooling breaks, resumable across shutdowns.
#
# One job at a time, a mandatory rest after each, and a thermal gate before the next -- because this is a
# laptop with a 4 GB card that runs at 73-74 C under sustained training load, and the plan is to work
# through the backlog over days rather than to finish it in one burn.
#
# What it guarantees, and why each guarantee exists:
#
#   SERIAL. Never two jobs at once. Two training runs do not fit in 4 GB anyway, and running a CPU
#   analysis beside a GPU run still adds heat to the same chassis.
#
#   REST AFTER, GATE BEFORE. Each step is followed by a fixed rest period, and the next step then waits
#   for the GPU to be idle and below a temperature threshold for a sustained window. Rest and gate are
#   separate on purpose: the rest is deliberate downtime regardless of readings, the gate is the
#   measurement. A card that has just finished a job reads cool for a moment before heat soaks out of
#   the heatsink, so a single cool sample is not evidence of anything.
#
#   RESUMABLE. Completed step NAMES are appended to a state file as they finish. Re-running the queue
#   skips them. So a shutdown -- planned, thermal, or a power cut -- costs at most the step that was in
#   flight, and that step's own training checkpoints (`last.pt`, saved every validation epoch) cover the
#   rest. Nothing has to be remembered by hand.
#
#   STOPS ON FAILURE. A failed step halts the queue and says so. An earlier chain in this project used
#   `set -e`, which did NOT abort it: three runs failed and it reported success. A queue that cannot fail
#   loudly is worse than no queue, because it produces a confident "done" with nothing behind it.
#
#   LEAVES RUNNING WORK ALONE. The gate waits for any python training process to exit, so launching this
#   while Arm B is mid-flight is safe: it simply waits its turn.
#
# Usage:
#   scripts/master_queue.sh status            # what is done, what is next
#   scripts/master_queue.sh run               # work the queue (foreground)
#   scripts/master_queue.sh start             # same, detached; survives closing the terminal
#   scripts/master_queue.sh stop              # stop the queue (does NOT kill a running step)
#   scripts/master_queue.sh reset <name>      # forget one completed step so it runs again
#
# Options (env vars):
#   REST=1200        seconds of rest after each step            (default 1200 = 20 min)
#   COOL_TO=55       start the next step below this GPU temp C   (default 55)
#   CPU_COOL_TO=80   ...and below this CPU package temp C         (default 80; 0 disables)
#   COOL_FOR=300     ...held for this many seconds continuously  (default 300 = 5 min)
#   ONLY=3           work at most N steps this session, then exit cleanly
#   GATE_MAX_WAIT=14400  give up on a step whose thermal gate never opens (default 4 h; 0 = forever)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

# Both overridable so the runner can be exercised against a throwaway queue without touching the real
# state -- which is how the fail and resume paths below were verified rather than assumed.
STATE_DIR="${QUEUE_STATE_DIR:-$REPO/logs_queue}"
STEPS_FILE="${QUEUE_STEPS:-$REPO/scripts/queue_steps.sh}"
STATE="$STATE_DIR/master_queue.done"
LOG="$STATE_DIR/master_queue.log"
PIDFILE="$STATE_DIR/master_queue.pid"
STOPFILE="$STATE_DIR/master_queue.stop"

REST="${REST:-1200}"
COOL_TO="${COOL_TO:-55}"
# The CPU is this laptop's real thermal bottleneck, not the GPU. Measured with only Arm B running and a
# load average of 1.3 out of 20 threads: GPU 72 C, x86_pkg_temp 87-94 C, TCPU/acpitz 92 C. Tjmax on an
# i7-12700H is 100 C, so the machine sits ~8 C from throttling on one busy dataloader thread. Gating on
# GPU temperature alone was watching the wrong sensor. 0 disables the CPU gate.
CPU_COOL_TO="${CPU_COOL_TO:-80}"
# How long a step may sit in the thermal gate before the queue gives up on it. Unbounded waiting is fine
# with a human watching, but left unattended it can deadlock: if this laptop's idle CPU temperature
# settles above CPU_COOL_TO, the gate would never open and nothing would say why. Four hours is far
# longer than any real cool-down and short enough that a stuck queue is noticed the same day. 0 restores
# the old unbounded behaviour.
GATE_MAX_WAIT="${GATE_MAX_WAIT:-14400}"
COOL_FOR="${COOL_FOR:-300}"
ONLY="${ONLY:-0}"
# Which processes a step must wait for. Overridable so the runner itself can be exercised while real
# training is in flight; operationally it stays at the default.
#
# CPU steps wait for GPU training too, deliberately: the brief is one job at a time with a rest between,
# and a CPU analysis still adds heat to the same chassis. The cost is that CPU work queues behind GPU
# work instead of filling the gaps, which roughly doubles wall-clock for a mixed backlog. CPU_PARALLEL=1
# lets cpu steps start while a GPU job runs, for days when throughput matters more than headroom.
GATE_WAIT="${GATE_WAIT:-train_surrogate_arm[.]py train_egnn[a-z_]*[.]py}"
CPU_PARALLEL="${CPU_PARALLEL:-0}"

mkdir -p "$STATE_DIR"
touch "$STATE"
# shellcheck source=queue_steps.sh
source "$STEPS_FILE"

log() { echo "[queue $(date '+%F %T')] $*"; }

done_already() { grep -Fxq "$1" "$STATE" 2>/dev/null; }

cmd_status() {
  local n_done=0 n_todo=0
  echo "queue: ${#STEPS[@]} step(s) defined"
  echo "state: $STATE"
  echo
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name kind _cmd <<< "$s"
    if done_already "$name"; then
      printf "  [done]  %-20s %s\n" "$name" "$kind"
      n_done=$((n_done + 1))
    else
      printf "  [ todo] %-20s %s\n" "$name" "$kind"
      n_todo=$((n_todo + 1))
    fi
  done
  echo
  echo "$n_done done, $n_todo remaining"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "queue is RUNNING (pid $(cat "$PIDFILE"))"
  else
    echo "queue is not running"
  fi
  local t ct
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  ct=$(for z in /sys/class/thermal/thermal_zone*/; do
         case "$(cat "$z/type" 2>/dev/null)" in
           x86_pkg_temp|TCPU|acpitz) cat "$z/temp" 2>/dev/null ;;
         esac
       done | sort -rn | head -1)
  [ -n "$ct" ] && ct=$(( ct / 1000 ))
  echo "GPU now: ${t:-?}C (gate <${COOL_TO}C)   CPU now: ${ct:-?}C (gate <${CPU_COOL_TO}C)"
  if [ -n "$ct" ] && [ "$ct" -ge "$CPU_COOL_TO" ]; then
    echo "  -> CPU is above its gate; the next step waits regardless of CPU_PARALLEL"
  fi
  if pgrep -f "train_.*[.]py" >/dev/null 2>&1; then
    echo "a training process is currently running -- the queue would wait for it"
  fi
  if [ -f "$STATE_DIR/master_queue.failed" ]; then
    echo "LAST RUN FAILED: $(cat "$STATE_DIR/master_queue.failed")"
    echo "  the queue stopped there on purpose; read logs_queue/step_<name>.log before re-running"
  fi
  if [ -f "$STATE_DIR/master_queue.gate_timeout" ]; then
    echo "LAST RUN TIMED OUT IN THE GATE: $(cat "$STATE_DIR/master_queue.gate_timeout")"
    echo "  nothing ran and nothing broke -- the machine stayed too hot for ${GATE_MAX_WAIT}s"
  fi
}

cmd_run() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "queue already running (pid $(cat "$PIDFILE")); use 'status' or 'stop'" >&2
    exit 1
  fi
  echo $$ > "$PIDFILE"
  rm -f "$STOPFILE"
  trap 'rm -f "$PIDFILE"' EXIT

  log "queue start -- rest ${REST}s after each step, gate <${COOL_TO}C for ${COOL_FOR}s before each"
  local worked=0 first=1
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name kind cmd <<< "$s"
    if done_already "$name"; then
      log "skip $name (already done)"
      continue
    fi
    if [ -f "$STOPFILE" ]; then
      log "stop requested -- leaving $name and the rest for next time"
      rm -f "$STOPFILE"
      break
    fi
    if [ "$ONLY" -gt 0 ] && [ "$worked" -ge "$ONLY" ]; then
      log "ONLY=$ONLY reached; $name and the rest left for next time"
      break
    fi

    # Rest BEFORE every step except the very first of this session: a session that begins right after a
    # previous one ended has already had its rest, and the gate below still applies either way.
    if [ "$first" -eq 0 ]; then
      log "resting ${REST}s before $name"
      sleep "$REST"
    fi
    first=0

    # Both kinds of step get a CPU thermal gate, because CPU heat is the binding constraint here and a
    # "CPU-only" analysis loads exactly the component that is already hot. A gpu step additionally
    # requires the card to be cool; a cpu step sets the GPU threshold high enough to be inert, since a
    # GPU at 72 C while the CPU is at 92 C should not be what holds up CPU work.
    #
    # CPU_PARALLEL=1 removes only the WAIT for other trainers, never the temperature gate. Asking for
    # parallel execution therefore does not override the measurement: a cpu step starts as soon as the
    # CPU has headroom, and the sensor decides when that is.
    local gate=() pat
    if [ "$kind" = "cpu" ]; then
      gate=(--cool-to 999 --cpu-cool-to "$CPU_COOL_TO" --cool-for "$COOL_FOR" --poll 60
            --max-wait "$GATE_MAX_WAIT")
      if [ "$CPU_PARALLEL" != "1" ]; then
        for pat in $GATE_WAIT; do gate+=(--wait-for "$pat"); done
      fi
    else
      gate=(--cool-to "$COOL_TO" --cpu-cool-to "$CPU_COOL_TO" --cool-for "$COOL_FOR"
            --poll 60 --max-wait "$GATE_MAX_WAIT")
      for pat in $GATE_WAIT; do gate+=(--wait-for "$pat"); done
    fi

    local steplog="$STATE_DIR/step_${name}.log"
    log "=== STEP $name ($kind) -> $steplog"
    # TQDM_DISABLE silences progress bars. Without it the queue log reached 18 MB after two training
    # steps, almost entirely a redrawn progress bar, which buries the per-epoch INFO lines that a status
    # check actually needs. The training scripts log every validation epoch anyway.
    # shellcheck disable=SC2086
    PYTHONPATH="$REPO" bash "$REPO/scripts/queued_run.sh" "${gate[@]}" --log "$steplog" -- \
      env PYTHONPATH="$REPO" TQDM_DISABLE=1 bash -c "$cmd"
    local rc=$?
    if [ $rc -eq 75 ]; then
      # EX_TEMPFAIL from queued_run: the start conditions never came true. Not a failure of the work.
      log "### STEP $name NEVER STARTED: the thermal gate did not open within ${GATE_MAX_WAIT}s."
      log "    Nothing ran and nothing is broken. The machine stayed too hot, or a wait pattern never"
      log "    cleared. Check the current readings with: scripts/master_queue.sh status"
      log "    If idle CPU temperature is simply above ${CPU_COOL_TO}C on this machine, raise the"
      log "    threshold once you know the real idle value, e.g. CPU_COOL_TO=85 ... start"
      echo "gate-timeout $name $(date '+%F %T')" > "$STATE_DIR/master_queue.gate_timeout"
      exit 75
    fi
    if [ $rc -ne 0 ]; then
      log "!!! STEP $name FAILED (exit $rc). Queue halted; nothing after it ran."
      log "    read $steplog, fix, then re-run: scripts/master_queue.sh run"
      echo "failed $name rc=$rc $(date '+%F %T')" > "$STATE_DIR/master_queue.failed"
      exit $rc
    fi
    rm -f "$STATE_DIR/master_queue.failed" "$STATE_DIR/master_queue.gate_timeout"
    echo "$name" >> "$STATE"
    worked=$((worked + 1))
    log "=== STEP $name done and recorded ($worked this session)"
  done
  log "queue idle -- $(grep -c . "$STATE" 2>/dev/null || echo 0) step(s) recorded as done in total"
}

case "${1:-status}" in
  status) cmd_status ;;
  run)    cmd_run ;;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))" >&2; exit 1
    fi
    setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
    disown
    sleep 2
    echo "queue started detached; log: $LOG"
    ;;
  stop)
    touch "$STOPFILE"
    echo "stop requested. The queue will finish the step it is on, then exit."
    echo "To interrupt the running step itself, kill it by PID -- its checkpoints make it resumable."
    ;;
  reset)
    [ $# -ge 2 ] || { echo "usage: $0 reset <step-name>" >&2; exit 2; }
    grep -Fxv "$2" "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    echo "forgot '$2'; it will run again"
    ;;
  *) echo "usage: $0 {status|run|start|stop|reset <name>}" >&2; exit 2 ;;
esac
