# Synth-Guidance Reinvestigation Log (Track D)

Documentation choice, stated explicitly per the Track C/D addendum: this
gets its own dedicated log (rather than folding into
`guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`, which is Track A/B's Stage 2+
phase) because Track D already has a rich existing document structure
(`SYNTH_GUIDANCE_FINDING.md`) that this log extends directly, keeping the
synthesizability side's full audit trail in one place.

| Step | Description | Result | Date | Links |
|---|---|---|---|---|
| D.2 | Status re-establishment: verified prior guidance-effect results were exploratory-scale (1 pocket, n=8, no statistical testing), the underlying synth model was capped at a fixed max_epochs=40 (not converged), and a previously-unflagged leakage-safety gap (no test set, no leakage check on the train/val split) | Confirmed all three gaps; recommended a leakage-safe re-split + full-convergence retrain (D.2a) before D.3's lambda re-sweep | 2026-09-12 | `guidance/TRACK_D_STATUS_D2.md` |
| D.2a | Leakage-safe re-split (reusing Stage 0's target-level assignment, no new RA-score computation) + full-convergence retrain | **Major correction**: real leakage-safe test R²=0.414, Pearson=0.647, Spearman=0.649 (vs. the original leakage-unchecked 0.769/0.878/0.871) -- early-stopped epoch 20, best epoch 5 | 2026-09-12 | `guidance/TRACK_D_LEAKAGE_SAFE_RETRAIN_FINDING.md`; `guidance/lp_split/build_synth_lp_splits.py`; `guidance/lp_split/train_synth_lp.py`; checkpoint `logs_synth_lp/synth_ra_egnn_lp_2026_09_12__15_38_21_full/checkpoints/best.pt` |
| D.3 | Full lambda re-sweep with bond-aware features, at proper power (n=30/lambda, single pocket), using the NEW leakage-safe checkpoint | Formally no signal after BH correction (min corrected p=0.997), but divergence direction persists: real RA-score lower than baseline at every non-zero lambda tested, and the classic own-score-up/real-score-down signature reappears at lambda=0.03 (N=26, own +5.7%, real -25.8%). Per D.4's gating rule, divergence NOT resolved -> skip D.4, proceed to D.5 | 2026-09-15 | `guidance/TRACK_D_D3_LAMBDA_RESWEEP_FINDING.md` |
| D.4 | Multi-pocket confirmation | **Skipped per D.4's explicit gating rule** -- D.3 showed the divergence persisting, not resolved, so multi-pocket confirmation effort is not warranted | 2026-09-15 | See D.3's finding doc for the gating rationale |
| D.5 | DIAG1-equivalent diagnostic for synth-guidance gradient direction (size/heaviness and aromaticity proxies, since SynthGuidance is ligand-only) | Neither proxy explains the gradient's misdirection: size corr=0.032 [-0.019,0.083], aromaticity corr=-0.014 [-0.061,0.036], both CI include 0 (n=144). Unlike DIAG1's affinity-side finding (clear distance-to-pocket confound), the synth-guidance mechanism's failure remains mechanistically unexplained | 2026-09-15 | `guidance/DIAG_SYNTH_DIRECTION_FINDING.md`; `guidance/diag_synth_direction_results.json` |

**Track D closed.** Overall conclusion: divergence persists after two independent fix attempts (bond-aware features, leakage-safe retrain) and remains mechanistically unexplained after two diagnostics (this one, plus the original SMILES-divergence check) -- see `guidance/DIAG_SYNTH_DIRECTION_FINDING.md`'s closing table for the full D.2-D.5 summary.
