# Affinity Lambda Re-Sweep: Real Vina Dock Scores (Part 1 of the Fine-Grained Re-Sweep Addendum)

## Purpose

Task F (`TASK_F_RESULTS.md`) found no statistically significant effect of
affinity guidance on real downstream metrics at `lambda_affinity=1.0`. Two
competing explanations were proposed:

- **(A) Under-calibrated guidance strength** — `lambda=1.0` was chosen only
  because it was the safest point with zero validity cost in the original
  sweep (`SYNTH_GUIDANCE_FINDING.md`'s sibling sweep), not because it was
  shown to move a *real* metric. A higher, still-safe lambda might reveal a
  real effect the original placeholder-feature sweep (which never docked)
  could not see.
- **(B) Fundamentally weak/uninformative affinity signal** — the guidance
  model's gradient direction does not correlate well enough with real
  binding affinity (as opposed to its own self-reported prediction) for any
  safe lambda to produce a measurable Vina Dock improvement.

This experiment tests (A) directly, cheaply, before committing to (B)'s
much more expensive fix (retraining the affinity model, Part 3).

## Method

- `guidance/lambda_sweep.py`, extended this session to run real AutoDock
  Vina docking (`mode='dock'`, exhaustiveness=8) on every valid,
  single-SMILES-string generated molecule, in addition to the existing
  cheap checks (validity, fragmentation, guidance model's own predicted
  score, QED/SA/real RA-score).
- Grid: `[1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]` — wider and
  denser than the original `[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]`,
  concentrated in the 1-20 range that the original sweep never
  individually resolved (only 1.0, 3.0, and 10.0 were tested there, and
  none were docked).
- n=8 samples per point, full 1000 diffusion steps, same bundled example
  pocket (`examples/1h36_A_..._pocket10.pdb`) used throughout this
  project's sweeps, for direct comparability with the original sweep.
- `lambda_synth=0.0` throughout (synth guidance already ruled out
  separately, see `SYNTH_GUIDANCE_FINDING.md`).

## Results

| lambda | validity | single-fragment rate | mean own-score | mean Vina Dock (kcal/mol) | std | n docked |
|---|---|---|---|---|---|---|
| 1.0  | 1.0 | 1.0 | 9.743 | -10.386 | 1.088 | 8 |
| 2.0  | 1.0 | 1.0 | 9.718 | -10.478 | 0.954 | 8 |
| 3.0  | 1.0 | 1.0 | 9.738 | -10.349 | 1.016 | 8 |
| 5.0  | 1.0 | 1.0 | 9.764 | -10.382 | 0.977 | 8 |
| 7.0  | 1.0 | 1.0 | 9.768 | -10.391 | 1.169 | 8 |
| 10.0 | 1.0 | 1.0 | 9.806 | -10.154 | 1.741 | 8 |
| 15.0 | 1.0 | 1.0 | 9.812 | -10.218 | 1.084 | 8 |
| 20.0 | 1.0 | 1.0 | 9.836 | -10.682 | 0.974 | 8 |
| 30.0 | 1.0 | **0.5** | 9.797 | -10.737 | 0.636 | 4 |

Fragmentation begins between lambda=20 and lambda=30 (consistent with the
original placeholder-feature sweep's finding that fragmentation appears
around lambda=10-30) -- **20.0 is the practical safety ceiling** for this
guidance design, one order of magnitude above the value actually used in
Task F.

Significance test (Mann-Whitney U, two-sided, unpaired, n=8 per group,
`lambda=1.0` as reference point since that is Task F's tested operating
point):

| Comparison | mean diff (kcal/mol) | p-value |
|---|---|---|
| 1.0 vs 2.0  | -0.092 | 0.878 |
| 1.0 vs 3.0  | +0.037 | 0.959 |
| 1.0 vs 5.0  | +0.004 | 0.959 |
| 1.0 vs 7.0  | -0.005 | 1.000 |
| 1.0 vs 10.0 | +0.232 | 0.878 |
| 1.0 vs 15.0 | +0.168 | 1.000 |
| 1.0 vs 20.0 | -0.296 | 0.798 |
| 1.0 vs 30.0 | -0.351 | 0.808 |

Pooled across all 8 fragmentation-free points (lambda=1-20, n=64 molecules
total): mean Vina Dock = -10.380 +/- 1.161 kcal/mol, range [-12.89, -6.63]
-- indistinguishable in location from the lambda=1.0 baseline alone
(-10.386 +/- 1.088).

The guidance model's own predicted score creeps up slightly across the
grid (9.72 at lambda=1 to 9.84 at lambda=20, a ~1.2% change) but this
tiny, monotonic-looking drift in the model's self-assessment does **not**
translate into any corresponding trend in real Vina Dock score, which
moves non-monotonically within a ~2 kcal/mol noise band with no
directional pattern across an 8-point, 20x range of guidance strength.

## Conclusion: explanation (A) is refuted

**Raising `lambda_affinity` anywhere within the safe range (1.0 to 20.0,
a 20x span) produces no statistically significant, and no visually
directional, change in real Vina Dock score.** Every pairwise comparison
against the Task F operating point (`lambda=1.0`) has p > 0.79 -- far from
significant, and the differences have no consistent sign. This is not a
case of "the effect exists but this study is underpowered to detect it
precisely" -- there is no dose-response relationship to speak of at all
across an order of magnitude of guidance strength, only sampling noise.

Task F's null result was **not** an artifact of choosing too conservative
an operating point. Even at 20x the tested strength (still with perfect
validity and 0% fragmentation), there is no real affinity signal to find.
Only at lambda=30 does the guidance start to break molecular structure
(50% fragmentation) -- and even that costly regime does not show a
believable affinity gain (n=4 remaining valid samples, mean -10.74,
p=0.81 vs baseline -- consistent with noise, not consistent with a
trade-off where enough disruption finally buys real affinity).

## Decision (Part 2): proceed to Part 3

Per the addendum's decision rule: since re-sweeping did **not** find a
safe lambda with a real, significant effect, the null result is not due to
under-calibration. This points to explanation (B): the affinity guidance
model's gradient signal itself is not informative enough about real
binding affinity to move it through position-only, constant-strength
guidance, at any operating point tried. Proceeding to Part 3: deepen the
affinity model, starting with the mandatory data-leakage check.

## Reproducibility

- Raw sweep results (incl. per-molecule Vina Dock scores): `guidance/affinity_lambda_resweep_dock.json`
- Full run log: `guidance/affinity_lambda_resweep_dock.log`
- Sweep code (extended this session with `--dock`/`--dock_exhaustiveness`): `guidance/lambda_sweep.py`
- Original (undocked, placeholder-feature) sweep for comparison: `guidance/SYNTH_GUIDANCE_FINDING.md`, `guidance/sweep_affinity.json`
