"""Verify that numbers quoted in the thesis still match the result files they came from.

Written after a verification sweep found that EVERY hand-assembled table in this thesis had drifted
from its source in some way, while every table copied from structured output was exact:

  * the compute-budget table carried 0.01 h rounding drift against the queue logs;
  * the RQ2 table labelled four rows "KS" although no KS test was ever computed for them;
  * the representation-ladder table mislabelled a model family, wrote "varies" for a known dimension,
    and mixed one row in from a different script than the one the table claims to report.

None of those changed a conclusion, and none would have been caught by the build: the document
rendered perfectly throughout. They are the failure mode this file exists for. Each entry below pins
one quoted number to the file and path it must agree with, so a future edit that retypes a value, or a
rerun that shifts one, fails here instead of reaching a reader.

Run:  PYTHONPATH=. python guidance/thesis/check_numbers_against_sources.py
Exit code 1 if any claim disagrees with its source.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CHAPTERS = os.path.join(HERE, 'chapters')


def norm_minus(s):
    """Treat the typographic minus the thesis uses (U+2212) and the ASCII hyphen a formatter emits as
    the same character. Without this the checker reports a false mismatch on every negative value."""
    return s.replace('\u2212', '-').replace('\u2010', '-').replace('\u2011', '-')


def jpath(obj, path):
    """Fetch a value by dotted path; list indices are written as integers.

    A path segment of "*" descends into a dict that holds exactly one entry, whatever its key. That is
    needed because some result files key their payload by a filesystem path ("./guidance_models/x.pt"),
    whose own dots would otherwise be read as further path separators.
    """
    cur = obj
    for part in path.split('.'):
        if part == '*':
            if not isinstance(cur, dict) or len(cur) != 1:
                raise KeyError(f'"*" needs a single-entry dict, got {type(cur).__name__} '
                               f'with {len(cur) if isinstance(cur, dict) else "n/a"} entries')
            cur = next(iter(cur.values()))
        else:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def load(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as f:
        return json.load(f)


# (label, source file, dotted path, formatter, chapter file, context)
# The formatter turns the source value into the exact string the thesis should contain. `context`, when
# given, must appear on the SAME LINE as the value: without it the check only asks whether the number
# occurs somewhere in the chapter, which a first version of this file did, and which silently passed
# when a table row was corrupted because the same figure also appeared in the surrounding prose.
CLAIMS = [
    # --- representation ladder (Chapter IV) -------------------------------------------------------
    ('ladder ligand R2',      'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.ligand.test_r2',          lambda v: f'{v:.4f}', '04_predictor_quality.txt', '| Ligand-only (14 RDKit'),
    ('ladder pocket R2',      'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.pocket.test_r2',          lambda v: f'{v:.4f}', '04_predictor_quality.txt', '| Pocket-only (38'),
    ('ladder ecif_raw R2',    'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.ecif_raw.test_r2',        lambda v: f'{v:.4f}', '04_predictor_quality.txt', '| ECIF, raw'),
    ('ladder ecif_norm R2',   'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.ecif_norm.test_r2',       lambda v: f'{v:.4f}', '04_predictor_quality.txt', '| ECIF, size-normalised'),
    ('ladder plip_raw R2',    'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.plip_raw.test_r2',        lambda v: f'{v:.4f}', '04_predictor_quality.txt', 'counts, raw |'),
    ('ladder plip_norm R2',   'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.plip_norm.test_r2',       lambda v: f'{v:.4f}', '04_predictor_quality.txt', 'counts, size-normalised |'),
    ('ladder plip+size R2',   'guidance/cheminformatics/representation_ladder_results.json',
     'blocks.plip_norm+size.test_r2',  lambda v: f'{v:.4f}', '04_predictor_quality.txt', 'size covariate'),
    # --- ligand-only ceiling and heavy-atom baseline (quoted throughout) --------------------------
    ('ligand-only ceiling',   'guidance/cheminformatics/ligand_only_results.json',
     'baselines.desc_ridge.test.R2',   lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('heavy-atom baseline',   'guidance/cheminformatics/ligand_only_results.json',
     'baselines.heavy_atoms.test.R2',  lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    # --- activity cliffs (Chapter IV) -------------------------------------------------------------
    ('cliff pairs',           'guidance/cheminformatics/activity_cliffs_results.json',
     'n_cliff_pairs',                  lambda v: f'{v:,}',   '04_predictor_quality.txt'),
    ('cliff compounds',       'guidance/cheminformatics/activity_cliffs_results.json',
     'n_cliff_compounds',              lambda v: f'{v:,}',   '04_predictor_quality.txt'),
    ('pairs checked',         'guidance/cheminformatics/activity_cliffs_results.json',
     'n_pairs_checked',                lambda v: f'{v:,}',   '04_predictor_quality.txt'),
    # --- class-prior decomposition and power analysis (Chapter IV.8), quoted throughout ----------
    ('chance (global y-rand)', 'guidance/cheminformatics/noise_ceiling_results.json',
     'decomposition.chance',               lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('class-level prior',      'guidance/cheminformatics/noise_ceiling_results.json',
     'decomposition.class_level_prior',    lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('within-target y-rand SD', 'guidance/cheminformatics/noise_ceiling_results.json',
     'y_randomisation.within_target.sd',   lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('label-noise ceiling',    'guidance/cheminformatics/noise_ceiling_results.json',
     'decomposition.ceiling',              lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('headroom beyond prior',  'guidance/cheminformatics/noise_ceiling_results.json',
     'decomposition.headroom_beyond_prior', lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('power window',           'guidance/cheminformatics/noise_ceiling_results.json',
     'power.window',                       lambda v: f'{v:.4f}', '04_predictor_quality.txt'),
    ('mean CI width',          'guidance/cheminformatics/noise_ceiling_results.json',
     'power.mean_ci_width',                lambda v: f'{v:.3f}', '04_predictor_quality.txt'),
    ('window / CI ratio',      'guidance/cheminformatics/noise_ceiling_results.json',
     'power.window_over_ci',               lambda v: f'{v:.2f}', '04_predictor_quality.txt'),
    ('ensemble % of headroom', 'guidance/cheminformatics/noise_ceiling_results.json',
     'models.egnn_a1c_ensemble.pct_of_headroom', lambda v: f'{v:.1f}%', '04_predictor_quality.txt'),
    ('desc_ridge % of headroom', 'guidance/cheminformatics/noise_ceiling_results.json',
     'models.desc_ridge (lig).pct_of_headroom',  lambda v: f'{v:.1f}%', '04_predictor_quality.txt'),
    ('vina % of headroom',     'guidance/cheminformatics/noise_ceiling_results.json',
     'models.vina (lig).pct_of_headroom',  lambda v: f'{v:.1f}%', '04_predictor_quality.txt'),
    # --- surrogate arms (Chapter IV) --------------------------------------------------------------
    ('Arm A mean R2',         'guidance/surrogate_data/armAC_table.json',
     'groups.surrogate_arms/armA_cd20000_bn0_bs1.mean',      lambda v: f'{v:.3f}', '04_predictor_quality.txt'),
    ('Arm C mean R2',         'guidance/surrogate_data/armAC_table.json',
     'groups.surrogate_arms/armC_cd6000_bn14000_bs1.mean',   lambda v: f'{v:.3f}', '04_predictor_quality.txt'),
    # --- M2 subgroup characterisation (Chapter VIII) ----------------------------------------------
    ('M2 complexes covered',  'guidance/track_e/m2_subgroup_characterisation.json',
     'n_complexes_included',           lambda v: f'{v:,}',   '08_track_e.txt'),
    ('M2 pct covered',        'guidance/track_e/m2_subgroup_characterisation.json',
     'pct_complexes_covered',          lambda v: f'{v:.1f}%', '08_track_e.txt'),
    ('M2 pk excluded',        'guidance/track_e/m2_subgroup_characterisation.json',
     'pk_mean_excluded',               lambda v: f'{v:.2f}', '08_track_e.txt'),
    ('M2 pk included',        'guidance/track_e/m2_subgroup_characterisation.json',
     'pk_mean_included',               lambda v: f'{v:.2f}', '08_track_e.txt'),
    # --- RQ2 positive control (Chapter VI), the thesis's most consequential table ----------------
    ('RQ2 vina_dock baseline', 'guidance/rq2_table.json',
     'metrics.vina_dock.baseline_mean',  lambda v: f'{v:.3f}', '06_track_c.txt', '| Vina Dock'),
    ('RQ2 vina_dock reranked', 'guidance/rq2_table.json',
     'metrics.vina_dock.reranked_mean',  lambda v: f'{v:.3f}', '06_track_c.txt', '| Vina Dock'),
    ('RQ2 sa reranked',        'guidance/rq2_table.json',
     'metrics.sa.reranked_mean',         lambda v: f'{v:.3f}', '06_track_c.txt', '| SA score'),
    ('RQ2 ra_score reranked',  'guidance/rq2_table.json',
     'metrics.ra_score.reranked_mean',   lambda v: f'{v:.3f}', '06_track_c.txt', '| RA-score'),
    ('RQ2 qed reranked',       'guidance/rq2_table.json',
     'metrics.qed.reranked_mean',        lambda v: f'{v:.3f}', '06_track_c.txt', '| QED'),
    ('RQ2 heavy_atoms base',   'guidance/rq2_table.json',
     'metrics.heavy_atoms.baseline_mean', lambda v: f'{v:.2f}', '06_track_c.txt', '| Heavy atoms'),
    ('RQ2 heavy_atoms rerank', 'guidance/rq2_table.json',
     'metrics.heavy_atoms.reranked_mean', lambda v: f'{v:.2f}', '06_track_c.txt', '| Heavy atoms'),
    ('RQ2 pb_valid reranked',  'guidance/rq2_table.json',
     'metrics.pb_valid.reranked_mean',   lambda v: f'{v:.3f}', '06_track_c.txt', '| PoseBusters'),
    # --- DIAG2 signed gradient direction (Chapter V) ----------------------------------------------
    ('DIAG2 n atom-states',   'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.n_atoms',
     lambda v: f'{v:,}', '05_tracks_ab.txt'),
    ('DIAG2 mean cosine',     'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.mean_cos_toward_contact',
     lambda v: f'{v:.3f}', '05_tracks_ab.txt'),
    ('DIAG2 frac toward',     'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.frac_pointing_toward',
     lambda v: f'{v:.1%}', '05_tracks_ab.txt'),
    ('DIAG2 near half',       'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.mean_cos_near_half',
     lambda v: f'{v:.3f}', '05_tracks_ab.txt'),
    ('DIAG2 far half',        'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.mean_cos_far_half',
     lambda v: f'{v:.3f}', '05_tracks_ab.txt'),
    ('DIAG2 fd frac increasing', 'guidance/diag_gradient_direction_results.json',
     'checkpoints.*.finite_difference_frac_increasing',
     lambda v: f'{v:.1%}', '05_tracks_ab.txt'),
    # --- compute budget (Appendix A) --------------------------------------------------------------
    ('Arm A total hours',     'guidance/surrogate_data/compute_budget_table.json',
     'totals.armA.wall_clock_h',       lambda v: f'{v:.2f}', '11_appendices.txt', '**Arm A total**'),
    ('Arm C total hours',     'guidance/surrogate_data/compute_budget_table.json',
     'totals.armC.wall_clock_h',       lambda v: f'{v:.2f}', '11_appendices.txt', '**Arm C total**'),
    ('Arm A total epochs',    'guidance/surrogate_data/compute_budget_table.json',
     'totals.armA.epochs',             lambda v: f'**{v}**',  '11_appendices.txt', '**Arm A total**'),
    ('Arm C total epochs',    'guidance/surrogate_data/compute_budget_table.json',
     'totals.armC.epochs',             lambda v: f'**{v}**',  '11_appendices.txt', '**Arm C total**'),
]


def check():
    cache, problems, checked = {}, [], 0
    for entry in CLAIMS:
        label, src, path, fmt, chapter = entry[:5]
        context = entry[5] if len(entry) > 5 else None
        try:
            if src not in cache:
                cache[src] = load(src)
            value = jpath(cache[src], path)
        except (OSError, KeyError, IndexError, ValueError) as e:
            problems.append(f'{label}: cannot read {src}:{path} ({type(e).__name__})')
            continue
        want = norm_minus(fmt(value))
        raw = norm_minus(open(os.path.join(CHAPTERS, chapter), encoding='utf-8').read())
        checked += 1
        if context:
            # the value must sit on the same line as its row label, so a single corrupted table cell
            # cannot hide behind the same number appearing elsewhere in the chapter
            hit = any(context in ln and want in ln for ln in raw.split('\n'))
            where = f'on a line containing "{context}"'
        else:
            hit = want in re.sub(r'\s+', ' ', raw)
            where = 'anywhere'
        if not hit:
            problems.append(f'{label}: source {src}:{path} = {value!r} -> expected "{want}" '
                            f'{where} in {chapter}, not found')
    return checked, problems


def main():
    checked, problems = check()
    if problems:
        print(f'NUMBER CHECK FAILED ({len(problems)} of {len(CLAIMS)} claims):')
        for p in problems:
            print('  -', p)
        return 1
    print(f'number check: {checked} quoted values all match their source files -- OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
