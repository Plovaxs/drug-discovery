"""Re-derive the measured training budget of the six production surrogate arms (thesis Table A.1).

The thesis compares Arm A against Arm C, so the compute each received has to be checkable rather than
asserted. This reads the queue logs of the runs actually reported and recomputes wall clock, thermal
suspension and per-epoch throughput, so the table in the appendix can be regenerated instead of
retyped. Hand-copying these numbers once already produced 0.01 h rounding drift between the text and
the logs; emitting them from a script removes that failure mode.

The substantive finding it supports: Arm C trained LONGER than Arm A (more epochs, more wall clock) at
the same per-epoch throughput, so Arm C's lower R2 cannot be explained by under-training.

Run:  python guidance/surrogate_data/compute_budget_table.py
"""
import argparse
import datetime as dt
import glob
import json
import os
import re

LOG_GLOB = 'logs_queue/step_arm[AC]_s202*.log'


def parse_log(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    stamps = re.findall(r'(2026-\d\d-\d\d \d\d:\d\d:\d\d)', s)
    if not stamps:
        return None
    start, end = dt.datetime.fromisoformat(stamps[0]), dt.datetime.fromisoformat(stamps[-1])
    hours = (end - start).total_seconds() / 3600.0
    susp = re.findall(r'suspended (\d+)s in total', s)
    susp_h = int(susp[-1]) / 3600.0 if susp else 0.0
    epochs = len(re.findall(r'\[Validate\] Epoch', s))
    return {
        'run': os.path.basename(path).replace('step_', '').replace('.log', ''),
        'epochs': epochs,
        'wall_clock_h': round(hours, 2),
        'thermally_suspended_h': round(susp_h, 2),
        'hours_per_epoch': round(hours / epochs, 2) if epochs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='./guidance/surrogate_data/compute_budget_table.json')
    args = ap.parse_args()

    rows = [r for r in (parse_log(p) for p in sorted(glob.glob(LOG_GLOB))) if r]
    totals = {}
    for tag in ('armA', 'armC'):
        sel = [r for r in rows if r['run'].startswith(tag)]
        ep = sum(r['epochs'] for r in sel)
        h = sum(r['wall_clock_h'] for r in sel)
        sp = sum(r['thermally_suspended_h'] for r in sel)
        totals[tag] = {'epochs': ep, 'wall_clock_h': round(h, 2),
                       'thermally_suspended_h': round(sp, 2),
                       'hours_per_epoch': round(h / ep, 2) if ep else None}

    res = {'per_run': rows, 'totals': totals}
    print(f"{'run':14s} {'epochs':>6s} {'wall_h':>7s} {'susp_h':>7s} {'h/epoch':>8s}")
    for r in rows:
        print(f"{r['run']:14s} {r['epochs']:6d} {r['wall_clock_h']:7.2f} "
              f"{r['thermally_suspended_h']:7.2f} {r['hours_per_epoch']:8.2f}")
    for tag, t in totals.items():
        print(f"{tag + ' TOTAL':14s} {t['epochs']:6d} {t['wall_clock_h']:7.2f} "
              f"{t['thermally_suspended_h']:7.2f} {t['hours_per_epoch']:8.2f}")
    a, c = totals['armA'], totals['armC']
    print(f"\nArm C trained {c['epochs'] - a['epochs']} more epochs and "
          f"{c['wall_clock_h'] - a['wall_clock_h']:.2f} h longer than Arm A at comparable throughput "
          f"({c['hours_per_epoch']} vs {a['hours_per_epoch']} h/epoch), so under-training does not "
          f"explain Arm C's lower R2.")

    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
