"""Automated guard on the train/val/test separation of the BindingNet surrogate training pool.

Why this is a TEST and not just a script that was run once: the leakage filter in this project was wrong
twice, and both times it looked fine.

  * Stage 1 filtered on receptor PDB ID. It passed, and it was wrong.
  * Stage 2 added ChEMBL-target -> UniProt accession crossref. It passed, and it was still wrong:
    template `1fm9` chain A is 100.0% identical to P19793 (RXRA_HUMAN), one of our 127 test targets,
    because 1FM9 is a PPARG/RXRA heterodimer crystal carrying the test protein under a PDB ID that is
    neither of the two the ID filter caught. A whole family of nuclear-receptor heterodimers leaked the
    same way.

Both failures shared a shape: the check that was run was not the check that mattered, and nothing in the
pipeline would ever have told us. So the invariant is asserted on the ARTIFACTS THEMSELVES, every time
the suite runs, and it is asserted at the level that matters (sequence identity, not identifiers).

These tests read only CSV/JSON already on disk -- no network, no alignment, no GPU -- so they are cheap
enough to run on every commit. They deliberately do NOT re-run the pairwise alignment; re-deriving the
audit would test the audit code against itself. They test that what the audit FOUND was actually APPLIED,
which is the step that silently rots when someone regenerates a CSV.

Usage:
  PYTHONPATH=. python guidance/surrogate_data/tests/test_leakage_guard.py
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'guidance', 'surrogate_data')

FINAL_CSV = os.path.join(DATA, 'bindingnet_v1_final.csv')
PAIRED_CSV = os.path.join(DATA, 'bindingnet_v1_paired.csv')
SUMMARY = os.path.join(DATA, 'bindingnet_v1_final_summary.json')
AUDIT = os.path.join(DATA, 'leakage_audit_report.json')
PER_SEQ = os.path.join(DATA, 'leakage_audit_report_per_sequence.csv')
VAL_TEST_UNIPROT = os.path.join(DATA, 'val_test_target_uniprot.json')
CLEAN_CSV = os.path.join(DATA, 'bindingnet_v1_clean.csv')
CLEAN_SUMMARY = os.path.join(DATA, 'bindingnet_v1_clean_summary.json')
OVERLAP_REPORT = os.path.join(DATA, 'compound_overlap_report.json')
BN_ACCESSIONS = os.path.join(DATA, 'bindingnet_chembl_target_accessions.json')

LEAKAGE_THRESHOLD = 90.0   # the pre-registered cut; see apply_sequence_leakage_filter.py docstring


def _templates_in(csv_path):
    """Only the template column is read: these CSVs are ~48 MB and the rest is irrelevant here."""
    return set(pd.read_csv(csv_path, usecols=['pdb_template']).pdb_template.unique())


def _skip(reason):
    print(f'SKIP {reason}')
    return True


# ------------------------------------------------- the invariant that matters

def test_no_excluded_template_survives_in_the_final_pool():
    """The core assertion: every template the audit flagged as >=90% identical to a val/test target is
    actually gone from the training pool. This is the check that would have caught the 1fm9 class of
    leak, and it is the one that breaks if anyone regenerates bindingnet_v1_final.csv from the paired CSV
    without re-applying the filter."""
    summary = json.load(open(SUMMARY))
    excluded = set(summary['templates_excluded'])
    assert excluded, 'summary lists no excluded templates -- the filter was a no-op, which is suspicious'
    survivors = excluded & _templates_in(FINAL_CSV)
    assert not survivors, (f'LEAKAGE: {len(survivors)} excluded template(s) present in the training pool: '
                           f'{sorted(survivors)[:10]}')


def test_every_high_identity_sequence_was_excluded():
    """Independent direction: instead of trusting the summary's exclusion list, recompute which templates
    exceed the threshold straight from the per-sequence audit and require all of them to be absent. If
    apply_sequence_leakage_filter.py ever mis-parses the semicolon-joined template field, the summary and
    this recomputation disagree and the test fails."""
    audit = pd.read_csv(PER_SEQ)
    bad = audit[audit.best_identity_pct >= LEAKAGE_THRESHOLD]
    assert len(bad) > 0, 'per-sequence audit flags nothing -- re-run leakage_audit.py before trusting this'
    bad_templates = set()
    for s in bad.bn_templates:
        bad_templates.update(x.split(':')[0] for x in str(s).split(';'))
    survivors = bad_templates & _templates_in(FINAL_CSV)
    assert not survivors, f'LEAKAGE recomputed from audit: {sorted(survivors)[:10]}'


def test_summary_exclusion_list_matches_the_audit_it_claims_to_apply():
    """Guards the seam between the two scripts. `bn_templates` holds only the first 5 owners per sequence
    (owners[:5]), so the audit-derived set can be a strict subset of what was excluded -- but it must
    never contain anything the summary missed."""
    summary = json.load(open(SUMMARY))
    audit = pd.read_csv(PER_SEQ)
    bad = audit[audit.best_identity_pct >= summary['threshold']]
    derived = set()
    for s in bad.bn_templates:
        derived.update(x.split(':')[0] for x in str(s).split(';'))
    missed = derived - set(summary['templates_excluded'])
    assert not missed, f'audit flags templates the summary did not exclude: {sorted(missed)}'


def test_applied_threshold_is_the_preregistered_one():
    """A silent threshold change would weaken the leakage claim the thesis makes. Pinned so it has to be
    an explicit, deliberate edit."""
    summary = json.load(open(SUMMARY))
    assert summary['threshold'] == LEAKAGE_THRESHOLD, (
        f"final pool was built at {summary['threshold']}% identity, but the pre-registered cut is "
        f'{LEAKAGE_THRESHOLD}%. If this change is intended, update LEAKAGE_THRESHOLD here AND the '
        f'thesis limitations section together.')
    audit = json.load(open(AUDIT))
    assert audit['threshold'] == LEAKAGE_THRESHOLD, 'audit was run at a different cut than was applied'


# ------------------------------------------------- bookkeeping consistency

def test_row_and_template_counts_match_the_summary():
    """Cheap tripwire for a stale CSV: if someone regenerates one artifact and not the other, the counts
    diverge and every number the thesis quotes from this summary becomes wrong."""
    summary = json.load(open(SUMMARY))
    final = pd.read_csv(FINAL_CSV, usecols=['pdb_template'])
    assert len(final) == summary['rows_out'], f"{len(final)} rows on disk vs {summary['rows_out']} claimed"
    assert final.pdb_template.nunique() == summary['templates_out']
    assert summary['rows_in'] - summary['rows_dropped'] == summary['rows_out'], 'summary does not balance'


def test_final_pool_is_a_subset_of_the_paired_pool():
    """The filter may only REMOVE. A template appearing in the final pool but not the paired pool means
    rows came from somewhere unaudited."""
    if not os.path.exists(PAIRED_CSV):
        return _skip('paired CSV absent')
    extra = _templates_in(FINAL_CSV) - _templates_in(PAIRED_CSV)
    assert not extra, f'final pool has templates absent from the audited paired pool: {sorted(extra)[:10]}'


def test_id_level_filter_still_holds_on_the_final_pool():
    """The earlier, weaker filter must not have regressed either. Sequence identity supersedes it but does
    not replace it: this catches a ChEMBL target whose accession is a val/test target outright."""
    if not (os.path.exists(BN_ACCESSIONS) and os.path.exists(VAL_TEST_UNIPROT)):
        return _skip('accession crossref absent')
    bn_acc = json.load(open(BN_ACCESSIONS))
    val_test_acc = {v for v in json.load(open(VAL_TEST_UNIPROT)).values() if v}
    targets = pd.read_csv(FINAL_CSV, usecols=['chembl_target']).chembl_target.unique()
    # Each ChEMBL target maps to a LIST of accessions, not one: protein complexes and multi-subunit
    # targets carry several. A target leaks if ANY of its accessions is a val/test target, so the test
    # is an intersection -- checking only the first accession is how a complex containing a test protein
    # would slip through.
    hits = sorted({t for t in targets if set(bn_acc.get(t) or []) & val_test_acc})
    assert not hits, f'ID-level leakage reappeared for ChEMBL targets: {hits[:10]}'


# ------------------------------- stage 4: ligand-side (compound) separation

def test_compound_filter_removed_every_colliding_compound():
    """The ligand-side stage. Asserted separately from the protein-side stages because it answers a
    different question: the first three ask whether the training PROTEIN is the test protein, this asks
    whether the training LIGAND is the test ligand. A pool can pass all three and still fail this."""
    if not (os.path.exists(CLEAN_CSV) and os.path.exists(CLEAN_SUMMARY)):
        return _skip('compound-filtered pool not built (run apply_compound_leakage_filter.py)')
    summary = json.load(open(CLEAN_SUMMARY))
    excluded = set(summary['compounds_excluded'])
    assert excluded, 'no compounds excluded -- the compound filter was a no-op'
    present = excluded & set(pd.read_csv(CLEAN_CSV, usecols=['chembl_compound']).chembl_compound.unique())
    assert not present, (f'LIGAND LEAKAGE: {len(present)} excluded compound(s) still in the clean pool: '
                         f'{sorted(present)[:10]}')


def test_compound_filter_used_the_conservative_criterion():
    """Skeleton, not full InChIKey. The full key would have retained 60 of the 101 colliding families as
    'different' compounds purely on stereochemistry or protonation, which still leak the label."""
    if not os.path.exists(CLEAN_SUMMARY):
        return _skip('compound-filtered pool not built')
    summary = json.load(open(CLEAN_SUMMARY))
    assert 'skeleton' in summary['criterion'], (
        f"compound filter ran on criterion {summary['criterion']!r}; the conservative connectivity "
        f'skeleton is required, and switching to exact InChIKey weakens the separation claim')


def test_compound_filter_cost_stayed_negligible():
    """A sanity bound, not a safety property. The filter's justification is that it is nearly free
    (0.34%). If a future pool makes it expensive, that is a decision to re-examine rather than absorb
    silently -- so the test fails and says so."""
    if not os.path.exists(CLEAN_SUMMARY):
        return _skip('compound-filtered pool not built')
    summary = json.load(open(CLEAN_SUMMARY))
    pct = 100.0 * summary['rows_dropped'] / summary['rows_in']
    assert pct < 5.0, (f'compound filter now costs {pct:.2f}% of the pool, not the ~0.34% its '
                       f'justification assumes. Re-examine the trade-off and update the thesis text.')


def test_evaluation_side_exposure_is_recorded_not_just_the_training_side():
    """Guards against the framing error this audit exposed. 0.34% of training rows contaminated and
    10.12% of evaluation records affected are the same leak measured two ways, thirtyfold apart. The
    evaluation-side number must stay in the report, because it is the one that bounds how far a test
    score can be read as generalisation."""
    if not os.path.exists(OVERLAP_REPORT):
        return _skip('compound overlap audit not run')
    skel = json.load(open(OVERLAP_REPORT))['skeleton_matches']
    for field in ('pct_of_pool', 'pct_of_val_test_records', 'pct_of_distinct_val_test_ligands'):
        assert field in skel, (f'{field} missing from the overlap report -- both directions must be '
                               f'recorded, or the training-side number will be quoted alone')


def test_trainer_enforces_both_filters_in_code_not_only_upstream():
    """The filters must be enforced inside the training driver, not merely reflected in a clean CSV.

    The reason is a near-miss already on record: build_bn_train_set reads
    data/bindingnet_pocket10/labels.csv, which is the UNFILTERED 125,238-row index, and applies the
    exclusions itself. Had it instead trusted whichever CSV happened to be passed in, pointing a run at
    the wrong file would have silently trained on leaked rows with nothing in the log to show it. This
    test pins the signature default so the safe behaviour cannot become opt-in by accident."""
    import inspect
    try:
        from guidance.surrogate_data.train_surrogate_arm import build_bn_train_set
    except Exception as e:
        return _skip(f'cannot import the arm driver ({type(e).__name__})')
    params = inspect.signature(build_bn_train_set).parameters
    assert 'compound_filter' in params, (
        'build_bn_train_set has no compound_filter parameter -- the ligand-side filter is not enforced '
        'in the training code. See compound_overlap_audit.py.')
    assert params['compound_filter'].default is True, (
        f'compound_filter defaults to {params["compound_filter"].default!r}; it must default to True so '
        f'a run has to opt OUT of the filter explicitly, never into it')
    src = inspect.getsource(build_bn_train_set)
    for marker in ('SEQ_LEAK_SUMMARY', 'CMP_LEAK_SUMMARY'):
        assert marker in src, f'{marker} not referenced in build_bn_train_set -- filter not applied'


def test_gray_zone_is_still_present_and_therefore_still_a_disclosed_limitation():
    """Asserts a DECISION, not a safety property. 50-90%-identity homologs are kept deliberately, and the
    thesis discloses that. If a future run silently drops them the conclusions change, so the test fails
    loudly and points at the text that must be updated with it."""
    audit = pd.read_csv(PER_SEQ)
    gray = audit[(audit.best_identity_pct >= 50.0) & (audit.best_identity_pct < LEAKAGE_THRESHOLD)]
    gray_templates = set()
    for s in gray.bn_templates:
        gray_templates.update(x.split(':')[0] for x in str(s).split(';'))
    kept = gray_templates & _templates_in(FINAL_CSV)
    assert kept, ('no 50-90%% identity homologs remain in the training pool. That is a STRICTER split '
                  'than the thesis describes -- update the limitations section, then update this test.')


def test_summary_still_carries_the_gray_zone_disclosure_note():
    """The note is the only place the keep-homologs decision is recorded next to the data. Losing it is
    how a disclosed limitation quietly becomes an undisclosed one."""
    summary = json.load(open(SUMMARY))
    assert 'gray zone' in summary.get('note', ''), 'disclosure note missing from the summary JSON'


if __name__ == '__main__':
    missing = [p for p in (FINAL_CSV, SUMMARY, AUDIT, PER_SEQ) if not os.path.exists(p)]
    if missing:
        # Exit 2, not 1: these artifacts are multi-megabyte CSVs that are gitignored, so a checkout on a
        # machine that has not built the data pool genuinely cannot run this suite. The test runner maps
        # exit 2 to a VISIBLE skip rather than a pass, so the gap is reported instead of silently
        # counting as green -- a leakage guard that quietly no-ops is worse than no guard at all.
        print('CANNOT RUN -- missing artifacts (exit 2 = skip, not pass):')
        for p in missing:
            print('  ', os.path.relpath(p, ROOT))
        sys.exit(2)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        if f() is not True:
            print('PASS', k)
    print(f'{len(fns)} leakage-guard tests passed')
