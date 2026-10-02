# Follow-up Phase, Pooled Multiple-Comparison Correction: 0/27 Significant

**Update 3:** 6 more tests added (kitchen-sink checkpoint -- all four
interventions combined -- prelambda exploratory tier,
λ∈{0.1,0.3,1.0,3.0,10.0,30.0}, see `KITCHENSINK_GUIDANCE_FINDING.md` --
another clean null, closing the ablation space). **Running total: 38
tests pooled, 0/38 significant after BH correction** (min raw p=0.0306
unchanged; min BH-corrected p=0.462).

**Update 2:** 5 more tests added (gradient-alignment checkpoint's
prelambda exploratory tier, λ∈{0.1,0.3,1.0,3.0,10.0}, see
`GRADALIGN_GUIDANCE_FINDING.md` -- a clean null with no encouraging
direction at any lambda, so no confirmatory tier was run for it), giving
32 tests, 0/32 significant at that point.

**Update 1:** originally pooled 25 tests (below); 2 more (ESM2-augmented
EGNN, confirmatory tier, λ=1.0 and λ=3.0, see `ESM2_GUIDANCE_FINDING.md`)
were added once that branch's confirmation run completed, giving 27
tests, 0/27 significant (min BH-corrected p=0.540 at that point). The
per-source breakdown and methodology below are otherwise unchanged from
the original 25-test pool.

The original four thesis tracks (A, B, C, D) each already carry their own
within-track Benjamini-Hochberg correction (`THESIS_FINDINGS_SYNTHESIS.md`
§6). This document does the same pooling exercise for the eight follow-up
branches run after that synthesis was written, several of which had a raw
test statistic computed at the time but no multiple-comparison correction
applied across them as a set — each was eyeballed individually ("diff is
small, p looks high") rather than formally pooled. This closes that gap.

## What is pooled, and what deliberately is not

**Pooled (25 tests, all real-docking comparisons from independently
trained follow-up checkpoints):**

| Source | Checkpoint | Mechanism | n tests |
|---|---|---|---|
| `guidance/track_a_followup_full_results/gradient_informativeness_results.csv` | GIGN+PIGNet2, noise-matched + vina-target | KS test, Vina Dock, n=30/cond | 16 |
| `guidance/lambda_sweep_noisematched_results.json` | EGNN, noise-matched | Paired Wilcoxon, n=8/cond | 3 |
| `guidance/lambda_sweep_vinatarget_results.json` | EGNN, vina-target | Paired Wilcoxon, n=8/cond | 3 |
| `guidance/lambda_sweep_vinatarget_softv_confirm_results.json` | EGNN, vina-target + soft-v | Paired Wilcoxon, n=15-16/cond (CONFIRMATORY tier only) | 2 |
| `guidance/ranker_pool_vinatarget_results.json` (reanalyzed, see below) | EGNN, vina-target, as ranker | Mann-Whitney, size-stratified top/bottom | 1 |

**Deliberately excluded:**
- `lambda_sweep_vinatarget_softv_highlam_results.json` (n=8 exploratory
  tier that *motivated* the confirm run above) — pooling an exploratory,
  hypothesis-generating batch together with its own confirmatory follow-up
  in one correction double-counts the same question and conflates two
  different inferential stages. It is reported on its own terms
  (a hypothesis that did not survive its own pre-specified confirmation),
  not folded into this table.
- The original Tracks A/B/C/D — already corrected within their own track
  in prior work; repooling them here would mix two different experimental
  phases run under different protocols.
- Gradient-alignment and ESM2 checkpoints — no real-docking test exists
  yet for either (see Pending Work below); nothing to pool.

## Ranker size-stratified test, independently re-verified

The carried-over figure for this test ("diff=-0.839, p=0.217") was
recomputed from scratch directly against `guidance/
ranker_pool_vinatarget_results.json` rather than trusted as-is, per this
project's standing verify-before-trusting rule. Method: 36 molecules with
both `n_heavy_atoms` and a valid `vina_dock` (4 of the pool's 40 lack one
or both fields, excluded), split into heavy-atom terciles, top/bottom
half by the ranker's own predicted affinity score within each tercile,
pooled across terciles (n=18 top, n=18 bottom), Mann-Whitney U.
**Reproduced exactly: mean difference -0.839 kcal/mol, p=0.2172.** The
number was correct; it had just never been pooled into a cross-branch
correction before now.

## Result

**0/25 significant after Benjamini-Hochberg correction.** Minimum raw
p-value: 0.0306 (Track A follow-up, CD38_HUMAN:λ=0.3); minimum
BH-corrected p-value: 0.719. Exactly 1 of 25 raw p-values falls below
0.05 — matching chance (0.05 × 25 ≈ 1.25 expected false positives under a
true null), not evidence of a real effect anywhere in the pool.

| | Value |
|---|---|
| Tests pooled | 25 |
| Significant, raw p<0.05 | 1/25 (4%, ≈ chance rate) |
| Significant, BH-corrected | 0/25 |
| Min raw p | 0.0306 |
| Min BH-corrected p | 0.719 |

## Why this matters for the thesis's central claim

This is not a new result — every individual test here was already
reported (mostly as "no significant effect" on inspection). What this
adds is the formal guarantee the thesis's own methodology requires: **no
claim of improvement without a statistical test surviving
multiple-comparison correction** (`THESIS_FINDINGS_SYNTHESIS.md` §1). That
standard was being met test-by-test informally; it is now met as a
single, explicit, pooled statement covering the entire follow-up phase at
once, closing the one remaining methodological gap between the original
four-track synthesis and the eight follow-up branches run after it.

Combined with `TRACK_A_FOLLOWUP_FULL_TIER_FINDING.md`'s running count,
the total independently-trained checkpoints showing zero significant,
correctly-directed, non-degraded real-docking effect is now 9 (3
architectures, 2 guidance mechanisms), and this document adds: **zero of
them survive even when every real-docking comparison across the entire
follow-up phase is pooled into one correction**, i.e. the null is not an
artifact of under-correction anywhere in this phase either.

## Pending work this does NOT cover

Gradient-alignment (checkpoint complete, test R²=0.345, Pearson=0.626,
best diagnostic align_corr=-0.75 of any variant tried) and the
ESM2-augmented EGNN (training in progress at time of writing, test
R²=0.396 already at epoch 2) have no real-docking test yet. Per every
prior checkpoint in this phase, diagnostic improvement predicting a real
docking effect would be the exception; it should not be assumed from this
document's pattern alone and will be tested directly, added to this pool
once available, and re-corrected as a whole rather than appended
piecemeal.
