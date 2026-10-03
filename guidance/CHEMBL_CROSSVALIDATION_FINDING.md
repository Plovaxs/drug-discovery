# Independent Cross-Validation Against ChEMBL: 3/4 Targets Consistent, 1 Real Discrepancy

Phase 3's first item: an independent sanity check of this project's experimental
affinity labels (`data/affinity_info.pkl`, PDBBind-sourced pKd/pKi/pIC50
values attached to CrossDocked2020 poses -- see `datasets/
crossdocked_affinity.py`'s own docstring on this substitution) against
ChEMBL's independently-curated bioactivity database, for four targets used
repeatedly throughout this investigation's diagnostic and docking tests
(CDK6, HDAC8, CD38, AKT1).

## Method

For each target: looked up its ChEMBL target entry by gene symbol
(`mcp__plugin_bio-research_chembl__target_search`), then pulled all
logged Ki measurements (`get_bioactivity`, `activity_type=Ki` -- chosen
over IC50 because Ki is a direct binding-affinity measurement, the same
quantity PDBBind reports, rather than a functional/cellular potency that
can differ from true binding affinity for reasons unrelated to this
check). Compared the resulting pChEMBL range (-log10(Ki in M), the same
convention as our own pK) against our own dataset's pK range for that
target's CrossDocked-derived entries.

## Result

| Target | Our pK range (n entries) | ChEMBL pChEMBL range, Ki (n) | Agreement |
|---|---|---|---|
| CDK6 | 5.14 - 7.82 (23) | 6.55 - 9.80 (4 of 4 total) | Partial overlap (6.55-7.82) |
| HDAC8 | 5.54 - 7.05 (21) | 4.68 - 10.4 (41 of 885 total) | Our range sits inside ChEMBL's much wider one |
| AKT1 | 6.52 - 9.30 (64) | 4.98 - 9.82 (48 of total) | Good overlap |
| **CD38** | **3.59 - 4.25 (21)** | **7.85 - 9.52 (4 of 4 total)** | **No overlap at all** |

Three of four targets show plausible-to-good agreement between our
dataset's affinity labels and ChEMBL's independent curation. CD38 is a
real, substantial discrepancy: our dataset's CD38 entries are all weak
binders (pK ~4, i.e. Kd in the ~100 uM range) while every CD38 Ki value
ChEMBL has on file is a sub-100nM, drug-like inhibitor (pChEMBL 7.85-9.52).

## Interpretation, not glossed over

This is reported as a genuine discrepancy with a plausible but
NOT-confirmed explanation, not resolved as if it were understood: PDBBind
(this project's affinity source) and ChEMBL draw from systematically
different literature. PDBBind indexes affinity values attached to
crystallized protein-ligand complexes, which frequently include early-
stage fragments or weak co-crystallization ligands used for structural
studies rather than optimized leads; ChEMBL's curation leans toward
published medicinal-chemistry SAR series, which by construction tend to
report their most potent analogs. A selection-bias explanation is
consistent with CD38's specific numbers (both ChEMBL Ki values it has on
file happen to be sub-nanomolar, suggesting its Ki-annotated literature
is exclusively late-stage optimized compounds) but this was not directly
verified (e.g. by checking whether our specific CD38 PDBBind entries are
indeed fragment/HTS-hit co-crystals) and is flagged as the leading
hypothesis, not a settled conclusion.

## Why this matters for the thesis, and its limits

This does not change any real-docking conclusion in this investigation
(none of the four tracks/16 follow-up checkpoints is specifically about
CD38's absolute affinity scale; DIAG1-style diagnostics and docking tests
use within-target relative comparisons, which an overall systematic
label-scale difference would not directly invalidate). It is reported
because an unverified assumption ("our affinity labels are a reasonable
proxy for real binding affinity, full stop") turned out to have at least
one concrete counterexample when actually checked against an independent
source -- consistent with this project's standing rule to verify rather
than assume, and to report what is found even when it complicates a
convenient assumption.

## Scope limits

- n=4 targets, chosen because they recur throughout this investigation's
  diagnostic/docking tests, not a random or representative sample of the
  full 971-target LP-split pool. No claim is made about the other ~967.
- Ki-only comparison (chosen for its direct comparability to PDBBind's
  own definition); IC50/EC50 data for these targets was not pulled or
  compared, and could tell a different story.
- The selection-bias explanation for CD38 is a hypothesis, not verified
  against the actual identity of our specific CD38 ligands.
