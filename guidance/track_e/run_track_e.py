"""Resumable Track E driver (Option F).

  python guidance/track_e/run_track_e.py status
  python guidance/track_e/run_track_e.py run --stage e1        # E1 arms + the 2 extra EGNN seeds, then STOP at the gate
  python guidance/track_e/run_track_e.py run --stage e2        # refuses to start unless runs_track_e/GATE_E1_PASSED exists

Safe to Ctrl+C / SIGTERM / suspend at any time: the signal is forwarded to the current training process, which writes an
atomic mid-epoch checkpoint and exits with code 75; re-running the same command resumes exactly there. A run is only
"done" once its DONE.json exists; the driver moves to the next run only then. Every event is appended to
runs_track_e/progress.jsonl (never overwritten). A thermal guard pauses training (via the same safe path) if the GPU
exceeds --hot_c and resumes once it cools below --cool_c.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.environ.get('TRACK_E_ROOT', './runs_track_e')
PROGRESS = os.path.join(ROOT, 'progress.jsonl')
PY = sys.executable
TRAIN = 'guidance/track_e/train_e.py'

GIGN = ['--family', 'gign']
E1_ARMS = [  # (arm name, extra args)
    ('abs_phys', ['--target', 'abs', '--head', 'phys']),      # A0': matched absolute-target control
    ('e1', ['--target', 'delta', '--head', 'phys']),          # E1 primary
    ('e1mlp', ['--target', 'delta', '--head', 'mlp']),        # readout ablation
]


def e1_stage():
    runs = []
    for seed in (1, 2, 3):                                     # seed-major so a pause leaves complete seed blocks
        for arm, extra in E1_ARMS:
            runs.append(dict(id=f'{arm}_s{seed}', args=['--arm', arm, '--seed', str(seed)] + GIGN + extra))
    for seed in (2022, 2023):                                  # 2 extra Stage-0 EGNN seeds (Option F / D4)
        runs.append(dict(id=f'egnn_s{seed}', args=['--arm', 'egnn', '--family', 'egnn', '--seed', str(seed)]))
    return runs


def best_e1_head():
    """Best E1 readout chosen on VALIDATION loss only (no test contact): lower seed-mean best val loss wins."""
    vals = {}
    for arm in ('e1', 'e1mlp'):
        v = []
        for seed in (1, 2, 3):
            p = os.path.join(ROOT, f'{arm}_s{seed}', 'train_complete.json')
            if os.path.exists(p):
                v.append(json.load(open(p))['best_val'])
        vals[arm] = sum(v) / len(v) if v else float('inf')
    return ('mlp' if vals['e1mlp'] < vals['e1'] else 'phys'), vals


def e2_stage():
    head, _ = best_e1_head()
    base = ['--family', 'gign', '--target', 'delta', '--head', head]
    runs = []
    for seed in (1, 2, 3):
        runs.append(dict(id=f'e1e2_s{seed}', args=['--arm', 'e1e2', '--seed', str(seed), '--aux', 'plip'] + base))
        runs.append(dict(id=f'e1e2scr_s{seed}', args=['--arm', 'e1e2scr', '--seed', str(seed), '--aux', 'scrambled'] + base))
    for lam in (0.1, 1.0):                                     # lambda sensitivity (single seed, sensitivity only)
        runs.append(dict(id=f'e1e2_lam{lam}_s1', args=['--arm', f'e1e2_lam{lam}', '--seed', '1', '--aux', 'plip', '--lam', str(lam)] + base))
    return runs                                                # conditional PCGrad arm is added by `pcgrad` stage after the cos check


def pcgrad_stage():
    head, _ = best_e1_head()
    base = ['--family', 'gign', '--target', 'delta', '--head', head]
    return [dict(id=f'e1e2pc_s{s}', args=['--arm', 'e1e2pc', '--seed', str(s), '--aux', 'plip', '--pcgrad'] + base) for s in (1, 2, 3)]


STAGES = {'e1': e1_stage, 'e2': e2_stage, 'pcgrad': pcgrad_stage}


def log_event(run_id, event, **kw):
    os.makedirs(ROOT, exist_ok=True)
    with open(PROGRESS, 'a') as f:
        f.write(json.dumps({'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': run_id, 'event': event, **kw}) + '\n')


def run_state(run_id):
    d = os.path.join(ROOT, run_id)
    if os.path.exists(os.path.join(d, 'DONE.json')):
        return 'done', json.load(open(os.path.join(d, 'DONE.json')))
    if os.path.exists(os.path.join(d, 'status.json')):
        s = json.load(open(os.path.join(d, 'status.json')))
        return 'in_progress', s
    return 'pending', {}


def gpu_temp():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], timeout=10)
        return int(out.decode().split()[0])
    except Exception:
        return None


def status():
    print(f"{'run':22s} {'state':12s} detail")
    for stage, fn in STAGES.items():
        try:
            runs = fn()
        except Exception:
            continue
        for r in runs:
            st, d = run_state(r['id'])
            if st == 'done':
                extra = f"test R2 {d['test_r2']:.3f} r {d['test_pearson']:.3f} best_epoch {d['best_epoch']}" + (f" aux_auroc {d['aux_auroc_macro']:.3f}" if 'aux_auroc_macro' in d else '')
            elif st == 'in_progress':
                extra = f"epoch {d.get('epoch')} step {d.get('step')} best_val {d.get('best_val'):.4f}"
            else:
                extra = ''
            print(f"{stage + '/' + r['id']:22s} {st:12s} {extra}")
    t = gpu_temp()
    print('GPU temp:', t, 'C')


def run_stage(stage, hot_c, cool_c, extra_child_args):
    if stage in ('e2', 'pcgrad') and not os.path.exists(os.path.join(ROOT, 'GATE_E1_PASSED')):
        print('REFUSING: E2 requires the E1 gate to be reviewed and passed (create runs_track_e/GATE_E1_PASSED after review).')
        return 2
    runs = STAGES[stage]()
    if stage == 'e2':
        head, vals = best_e1_head()
        log_event('-', 'e2_head_choice', head=head, val_loss_by_arm=vals)
    interrupted = {'flag': False}
    child = {'p': None}

    def forward(sig, frm):
        interrupted['flag'] = True
        if child['p'] is not None and child['p'].poll() is None:
            child['p'].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    for r in runs:
        if interrupted['flag']:
            return 75
        state, _ = run_state(r['id'])
        if state == 'done':
            continue
        while True:
            log_event(r['id'], 'resume' if state == 'in_progress' else 'start')
            child['p'] = subprocess.Popen([PY, TRAIN] + r['args'] + ['--run_root', ROOT] + extra_child_args, env={**os.environ, 'PYTHONPATH': '.'})
            paused_for_heat = False
            last_check = 0
            while child['p'].poll() is None:
                time.sleep(2)
                if time.time() - last_check > 30:
                    last_check = time.time()
                    t = gpu_temp()
                    if t is not None and t >= hot_c:
                        log_event(r['id'], 'thermal_pause', temp_c=t)
                        paused_for_heat = True
                        child['p'].send_signal(signal.SIGTERM)
            code = child['p'].returncode
            if code == 0:
                _, d = run_state(r['id'])
                log_event(r['id'], 'done', **{k: d[k] for k in ('test_r2', 'test_pearson', 'best_epoch') if k in d})
                break
            if code == 75 and paused_for_heat and not interrupted['flag']:
                while (gpu_temp() or 0) > cool_c and not interrupted['flag']:
                    time.sleep(20)
                if interrupted['flag']:
                    return 75
                state = 'in_progress'
                continue
            if code == 75 or interrupted['flag']:
                log_event(r['id'], 'interrupted')
                print(f"interrupted during {r['id']}; re-run the same command to resume")
                return 75
            log_event(r['id'], 'failed', exit_code=code)
            print(f"FAILED {r['id']} (exit {code}); stopping so nothing is silently skipped")
            return 1
    log_event('-', f'stage_{stage}_complete')
    if stage == 'e1':
        print('E1 STAGE COMPLETE. STOP: run analysis and review the E1 gate before any E2 run.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['status', 'run'])
    ap.add_argument('--stage', choices=list(STAGES), default='e1')
    ap.add_argument('--hot_c', type=int, default=86)
    ap.add_argument('--cool_c', type=int, default=75)
    a, rest = ap.parse_known_args()
    if a.cmd == 'status':
        status()
    else:
        sys.exit(run_stage(a.stage, a.hot_c, a.cool_c, rest))
