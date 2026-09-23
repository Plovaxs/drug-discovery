# Track D, Step D.2: Status Re-Establishment (verified against logs, not assumed)

Per the addendum's explicit instruction to verify rather than assume
before designing D.3, here is what was actually found on inspection.

## 1. Prior guidance-effect results are exploratory-scale only

`SYNTH_GUIDANCE_FINDING.md`'s both checkpoints (placeholder-feature
original, and Phase 1's bond-aware-feature fix attempt) were run via
`guidance/lambda_sweep.py --stage synth[--use_bond_aware]` on:
- **1 pocket only** (the bundled example, `examples/1h36_..._pocket10.pdb`)
  — never the diverse multi-pocket set used throughout the affinity
  investigation.
- **n=8 samples/lambda point**, single seed (42).
- **No statistical testing at all** — no KS test, no Benjamini-Hochberg
  correction, no bootstrap CIs. Conclusions were drawn from raw percentage
  tables and a per-molecule direction count (e.g. "6/8 samples degrade").

Both checkpoints' conclusions ("null/negative," "does not resolve the
divergence") are very likely *directionally* correct given how stark the
observed effects were (guidance's own score reliably rising while real
RA-score reliably fell, well beyond what 8-sample noise would produce),
but neither was tested at anything resembling the rigor now required.
**This confirms the addendum's suspicion exactly**: exploratory-scale
work that needs the same full-tier treatment Track A received.

## 2. The underlying synth model itself was NOT trained to genuine convergence

`guidance_models/synth_ra_score.pt` (`logs_synth/synth_ra_egnn_2026_09_04__01_46_21/`):
config `max_epochs: 40`, `early_stop_patience: 6`. Training ran to
**exactly epoch 40 and stopped** with patience counter at **1/6** — i.e.
it hit a fixed epoch ceiling, not a patience-exhausted stopping point.
Best checkpoint (epoch 39, val loss 0.395) is only one epoch before the
cutoff; whether the model had genuinely plateaued or was still on an
improving trajectory when training was cut off is unverified. This is
the same category of gap Track A's exploratory tier had (partial,
capped training) and needs the same fix (train with a much higher
epoch ceiling, let patience actually exhaust).

Current predictive quality (on the model's own validation set — see next
point for why this number needs an asterisk): RMSE=0.192, R²=0.769,
Pearson=0.878, Spearman=0.871. This is materially stronger correlation
than either affinity model ever achieved (Pearson ~0.58-0.61) — RA-score
is a much easier target to predict from 3D structure than experimental
binding affinity, which is itself informative context for interpreting
whatever Track D's guidance-effect result turns out to be.

## 3. A genuine leakage-safety gap, not previously flagged: no test set, no leakage check

`datasets/synth_dataset.py`'s `build_synth_splits()` does a **plain
random 90/10 train/val split** directly on the RA-score-labeled index
pool (`data/synth_ra_labels.pkl`, 15,000 entries sampled from the full
166,500-entry base dataset) — **no held-out test set at all**, and **no
leakage safety check** (the exact category of problem Stage 0 was built
to fix on the affinity side: the same protein pocket, or a near-identical
ligand, can appear in both the training and validation split here,
inflating the reported Pearson/R² above by an unknown amount). Every
number in `SYNTH_GUIDANCE_FINDING.md` and the checkpoint above rests on
this un-audited split.

**This was not something the addendum anticipated needing fixed** (it
assumed "train to full convergence" was the main gap), but per the
publication-grade standard now in force (P2, P4), evaluating Track D's
"resolved or not" question on a leakage-unchecked, test-set-free split
would not hold up to the same scrutiny the affinity side's Stage 0 work
was built specifically to survive.

**Good news: this is fixable without any new RA-score computation.**
`data/synth_ra_labels.pkl`'s 15,000 indices can be re-partitioned using
the EXISTING leakage-safe target-level split
(`guidance/lp_split/leakage_safe_split.json`'s `train_targets`/
`val_targets`/`test_targets`, already vetted for the exact
category/ligand/pocket-similarity leakage this needs) by looking up each
labeled index's target via `base[idx].ligand_filename.split('/')[0]` --
the same parsing `guidance/lp_split/build_split.py` already uses. Verified
by direct computation:

| Split | N (RA-score-labeled entries) |
|---|---|
| Train | 8,835 |
| Val | 1,015 |
| **Test (new — did not exist before)** | **1,940** |
| Discarded (target not in leakage-safe split) | 3,210 |

This gives a properly-sized, leakage-safe, genuinely held-out test set
for the first time on the synthesizability side.

## Recommended plan (before proceeding to D.3's lambda re-sweep)

1. **D.2a (new, small prerequisite)**: build `guidance/lp_split/
   build_synth_lp_splits.py` (mirrors `build_lp_splits` exactly, just
   keyed on RA-score labels instead of pk), and retrain the synth model
   to genuine convergence (high max_epochs ceiling, real patience
   exhaustion) on this leakage-safe split — this is a "Stage-0-equivalent"
   for the synthesizability side, reusing 100% of the already-built
   leakage-safety infrastructure, no new RA-score computation needed.
   Report the new, leakage-safe-and-tested predictive quality (with
   bootstrap CIs, per P2/P4) before touching the guidance/lambda-sweep
   question at all.
2. Only then proceed to D.3 (full lambda re-sweep, bond-aware features,
   multi-pocket, statistically rigorous) on top of this properly-
   validated backbone.

This adds a real but bounded and infrastructure-reuse-heavy step ahead of
D.3 — flagging it now (per this project's standing practice of surfacing
scope changes before absorbing them silently) rather than either skipping
it (which would undermine the publication-grade standard) or proceeding
without checking in.
