"""Structural figures for the chapters that carried no graphics at all.

Chapters I, II and X were entirely textual: between them they held roughly 9,200 words, two large
tables and not one figure. The figures here are structural rather than numerical, so the usual
protection in this project, drawing every number from a committed result file, does not apply in
the same form. A different protection is used instead.

**The two table-derived figures parse the thesis source itself.** `fig_thesis_map` reads the
`@tab objmap` block out of `chapters/01_introduction.txt` and `fig_literature_matrix` reads the
`@tab lit` block out of `chapters/02_literature.txt`. Nothing about either table is retyped here.
Each figure then asserts that every row it parsed was matched by exactly one entry in its own
layout map, and that every layout entry was used; if a row is added, removed or reworded in the
chapter, the figure build fails loudly instead of silently drawing a stale picture. That is the
same failure mode the number checker exists to prevent, applied to structure instead of to digits.

`fig_scope_boundary` is the exception: it is a conceptual diagram with no table behind it, and it
is drawn from a list declared in this file. It is included because Chapter X states its own scope
limits in prose that is easy to read past, and a reader who takes a null result out of its stated
bounds misreads the whole thesis. It claims nothing the chapter does not already claim in words.

Run:  PYTHONPATH=. python guidance/thesis/make_figures_chapters.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from thesis_tables import classify, match_rows, read_table

OUT = 'guidance/thesis/figures'
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY, DARK, GREEN, RED = '#2a78d6', '#eb6834', '#8a8f98', '#222222', '#2e8b57', '#c0392b'
PURPLE = '#7a4fa3'
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 200, 'savefig.dpi': 300})


# --------------------------------------------------------------------------- source-parsing helpers

def box(ax, x, y, w, h, text, face, edge, fontsize=7.2, textcolor=DARK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                facecolor=face, edgecolor=edge, linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, zorder=4, linespacing=1.35, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11,
                                 color=color, lw=lw, zorder=2,
                                 shrinkA=0, shrinkB=0))


def blank_axes(fig_w, fig_h):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.0); ax.axis('off')
    return fig, ax


# ------------------------------------------------------------------------------------- Chapter I

#: Verdict vocabulary. The thesis reports several distinct kinds of negative result and one
#: positive one, and collapsing them into "failed" would misrepresent it: a falsified
#: pre-registered hypothesis, an effect that proved to be an artefact, and a test that simply
#: found nothing are three different epistemic outcomes.
VERDICTS = {
    'null':      (GRAY,   'no effect detected'),
    'artefact':  (ORANGE, 'effect found, then explained away as an artefact'),
    'falsified': (RED,    'pre-registered hypothesis falsified'),
    'validated': (GREEN,  'positive control succeeded'),
    'mechanism': (BLUE,   'mechanism characterised'),
}

#: stage -> column. Each entry: probe substring of the objective cell, short label, verdict key.
THESIS_MAP = {
    # Stage 1: is there anything in the predictor to guide with?
    'ablation':  ('Does gradient guidance with the original predictor', 'Baseline ablation\nCh. IV', 'null', 0),
    'ladder':    ('What information does the predictor actually use',   'Representation ladder\nCh. IV.6', 'mechanism', 0),
    'bindingnet':('augmenting training data with a larger external',    'BindingNet augmentation\nCh. IV.7', 'null', 0),
    'prior':     ('How much of the apparent performance is a class-level', 'Class-prior decomposition\nCh. IV.8', 'mechanism', 0),
    # Stage 2: the gradient route
    'trackA':    ("Does the predictor's architecture limit",            'Track A: architecture\nCh. V', 'null', 1),
    'trackB':    ('Does the guidance mechanism (scale, objective',      'Track B: mechanism\nCh. V', 'null', 1),
    'diag':      ('What does the affinity gradient actually encode',    'Gradient diagnostics\nCh. V.4', 'mechanism', 1),
    'phase1':    ('Does directly repairing the diagnosed gradient',     'Repairing the diagnosis\nApp. E.1', 'null', 1),
    'perstep':   ('What happens, mechanistically,', 'Per-step trajectory\nApp. E.5', 'mechanism', 1),
    # Stage 3: the selection route
    'trackC':    ('Can non-gradient, post-hoc selection succeed',       'Track C: rejection sampling\nCh. VI', 'artefact', 2),
    'control':   ('Does a genuine, non-surrogate signal succeed',       'Positive control\nCh. VI.5', 'validated', 2),
    # Stage 4: redesigns and robustness
    'trackD':    ('Does the same guidance mechanism work for synthesizability', 'Track D: synthesizability\nCh. VII', 'null', 3),
    'trackE':    ('Does a residual (delta) training target',            'Track E: delta-learning\nCh. VIII', 'falsified', 3),
    'phase2':    ('Does the null survive independent, non-docking',     'Robustness of the null\nApp. E.2', 'null', 3),
}

STAGES = ['Is the signal\nany good?',
          'Does the gradient\nroute work?',
          'Does post-hoc\nselection work?',
          'Do redesigns and\nrobustness checks\nchange it?']


def fig_thesis_map():
    """Visual abstract: every research objective of {tab:objmap}, its chapter and its verdict.

    Parsed from `chapters/01_introduction.txt`; see the module docstring for why.
    """
    _, rows = read_table('01_introduction.txt', 'objmap')
    pairs = match_rows(rows, THESIS_MAP, col=0, what='thesis map')

    cols = {}
    for key, _row in pairs:
        _probe, label, verdict, stage = THESIS_MAP[key]
        cols.setdefault(stage, []).append((label, verdict))

    n_stage = len(STAGES)
    tallest = max(len(v) for v in cols.values())

    fig, ax = blank_axes(7.8, 6.0)
    x_pad, gap = 0.03, 0.02
    w = (1 - 2 * x_pad - gap * (n_stage - 1)) / n_stage
    title_y, head_y, head_h = 0.985, 0.80, 0.078
    bh, bgap = 0.068, 0.036
    first_y = head_y - 0.045 - bh

    ax.text(0.5, title_y, 'Does a learned guidance signal improve the molecules of a frozen\n'
                          'diffusion generator, judged by independently computed outcomes?',
            ha='center', va='top', fontsize=9.4, color=DARK, weight='bold', linespacing=1.45)

    for s_i in range(n_stage):
        x = x_pad + s_i * (w + gap)
        box(ax, x, head_y, w, head_h, STAGES[s_i], '#e8eef6', '#9fb0c6', fontsize=7.9, weight='bold')
        if s_i < n_stage - 1:
            arrow(ax, x + w + 0.0035, head_y + head_h / 2,
                  x + w + gap - 0.0035, head_y + head_h / 2, '#9fb0c6', lw=1.3)
        y = first_y
        for label, verdict in cols.get(s_i, []):
            colour = VERDICTS[verdict][0]
            box(ax, x, y, w, bh, label, colour + '1f', colour, fontsize=7.1)
            y -= bh + bgap

    # The counts are derived from the verdict map, never typed in, so the summary sentence cannot
    # drift from the boxes drawn above it.
    verdicts = [v for col in cols.values() for _lab, v in col]
    n_total = len(verdicts)
    n_success = sum(1 for v in verdicts if v == 'validated')
    n_diag = sum(1 for v in verdicts if v == 'mechanism')

    band_top = first_y - tallest * (bh + bgap) + bgap - 0.02
    ax.text(x_pad, band_top, 'Outcome of each experiment', fontsize=8.2, color=DARK,
            weight='bold', va='top')
    for i, (_key, (colour, meaning)) in enumerate(VERDICTS.items()):
        yy = band_top - 0.045 - i * 0.032
        ax.add_patch(FancyBboxPatch((x_pad, yy - 0.009), 0.024, 0.018,
                                    boxstyle='round,pad=0.002,rounding_size=0.004',
                                    facecolor=colour + '1f', edgecolor=colour, lw=1.0, zorder=3))
        ax.text(x_pad + 0.036, yy, meaning, fontsize=7.3, va='center', color=DARK)

    box(ax, 0.45, band_top - 0.200, 0.52, 0.190,
        f'Of the {n_total} experiments, {n_success} produced a defensible improvement:\n'
        'the positive control, which uses no trained surrogate. It validates the\n'
        'selection harness and localises the failure to the information content\n'
        f'of the signal rather than to an insensitive evaluation. A further {n_diag}\n'
        'characterise a mechanism rather than test for an improvement.',
        '#f2f7f3', GREEN, fontsize=7.4)

    fig.savefig(f'{OUT}/fig_thesis_map.png', bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------------------------ Chapter II

#: Probe substring of the "Work" cell -> short display name. Order here is the display order.
LIT_WORKS = {
    'DeepICL':   ('DeepICL', 'DeepICL'),
    'InterDiff': ('InterDiff', 'InterDiff'),
    'BInD':      ('BInD', 'BInD'),
    'DPDiff':    ('DPDiff', 'DPDiff'),
    'BADGER':    ('BADGER', 'BADGER'),
    'DiffGui':   ('DiffGui', 'DiffGui'),
    'BoKDiff':   ('BoKDiff', 'BoKDiff'),
    'PLANET':    ('PLANET', 'PLANET'),
    'VinaRF20':  ('VinaRF20', '\u0394VinaRF20 / \u0394-XGBoost'),
    'This thesis': ('This thesis', 'This thesis'),
}

STATE_STYLE = {
    'yes':     (GREEN, '●', 'present'),
    'no':      ('#ffffff', '–', 'absent'),
    'unknown': (GRAY, '○', 'not stated in the material reviewed'),
}

def fig_literature_matrix():
    """The gap of {tab:lit} as a matrix, parsed from `chapters/02_literature.txt`."""
    header, rows = read_table('02_literature.txt', 'lit')
    pairs = match_rows(rows, LIT_WORKS, col=0, what='literature matrix')

    crit = [h for h in header[1:]]
    short = ['3D diffusion\nSBDD',
             'Separate learned\nguidance predictor',
             'Interaction\ninformation',
             'Residual (delta)\ntarget',
             'Leakage-safe,\ntarget-disjoint eval.',
             'Independent eval.\nwith controls']
    if len(crit) != len(short):
        raise SystemExit(f'literature matrix: table has {len(crit)} criteria, figure labels {len(short)}')

    names = [LIT_WORKS[k][1] for k, _ in pairs]
    grid = [[classify(c) for c in row[1:]] for _k, row in pairs]
    if names[-1] != 'This thesis':
        raise SystemExit('literature matrix: expected the self-row last in {tab:lit}, '
                         f'found {names[-1]!r}')

    n_r, n_c = len(names), len(short)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(-0.5, n_c - 0.5); ax.set_ylim(-0.5, n_r - 0.5)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_r):
        if i == n_r - 1:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n_c, 1.0, facecolor='#eef5ef',
                                       edgecolor='none', zorder=0))
        for j in range(n_c):
            state = grid[i][j]
            colour = STATE_STYLE[state][0]
            # Markers, not Circle patches: a patch is drawn in data coordinates and comes out as an
            # ellipse whenever the axes aspect is not 1, which it is not here.
            ax.scatter([j], [i], s=190,
                       facecolor='#ffffff' if state == 'no' else colour,
                       edgecolor=DARK if state == 'no' else colour, linewidths=1.0, zorder=3)
            if state == 'no':
                ax.text(j, i, '–', ha='center', va='center', fontsize=8.5, color=DARK, zorder=4)

    ax.set_xticks(range(n_c)); ax.set_xticklabels(short, fontsize=7.4)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(names, fontsize=8)
    for lab in ax.get_yticklabels():
        if lab.get_text() == 'This thesis':
            lab.set_weight('bold')
    ax.tick_params(length=0)

    handles = [plt.Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=GREEN,
                          markeredgecolor=GREEN, label='present'),
               plt.Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor='white',
                          markeredgecolor=DARK, label='absent'),
               plt.Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=GRAY,
                          markeredgecolor=GRAY, label='not stated in the material reviewed')]
    ax.legend(handles=handles, fontsize=7.2, frameon=False, ncol=3,
              loc='upper center', bbox_to_anchor=(0.5, -0.06))
    fig.savefig(f'{OUT}/fig_literature_matrix.png', bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------------------------- Chapter X

INSIDE = [
    'One generator: a frozen, pocket-conditioned TargetDiff',
    'One data set: CrossDocked2020, with cross-docked poses',
    'Position-only gradients on the clean-data estimate',
    'Guidance strengths spanning three orders of magnitude',
    'Predictors trained on clean poses, not on noised ones',
    'Pocket sets of 1 to 20; 127 targets for predictor tests',
    'Evaluation by docking, PoseBusters, RA-score',
    'A single consumer GPU; tens of GPU-hours per arm',
]

OUTSIDE = [
    'Other generators, and non-frozen or jointly trained ones',
    'Noise-aware predictors trained on the sampler’s own states',
    'Gradients on atom types or bonds, not only on positions',
    'Larger pocket sets, which would detect smaller effects',
    'Wet-laboratory validation of any generated molecule',
    'Guidance signals that are not learned surrogates at all',
]


def fig_scope_boundary():
    """What the null result does and does not cover (Chapter X states both in prose)."""
    fig, ax = blank_axes(7.8, 4.0)
    ax.text(0.5, 1.0, 'What the null result of this thesis covers, and what it does not',
            ha='center', va='top', fontsize=9.6, color=DARK, weight='bold')

    top, line_h = 0.885, 0.072
    # Both panels get the height of the longer one, so the frames align; a reader should not have
    # to work out whether a shorter box means a shorter list or a weaker claim.
    h = 0.085 + line_h * max(len(INSIDE), len(OUTSIDE))
    ax.set_ylim(top - h - 0.16, 1.02)
    for x0, w, title, items, colour, note in (
            (0.02, 0.47, 'Tested here \u2014 the null applies', INSIDE, BLUE,
             'Within these bounds, no route built on a trained\nsurrogate produced a defensible improvement.'),
            (0.51, 0.47, 'Not tested \u2014 the null does not apply', OUTSIDE, ORANGE,
             'This thesis does not establish that learned guidance\nfails in general, and does not claim to.')):
        y0 = top - h
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle='round,pad=0.006,rounding_size=0.018',
                                    facecolor=colour + '12', edgecolor=colour, lw=1.3, zorder=1))
        ax.text(x0 + w / 2, top - 0.035, title, ha='center', va='center', fontsize=8.6,
                color=colour, weight='bold', zorder=4)
        y = top - 0.078
        for it in items:
            ax.text(x0 + 0.018, y, '\u2022', fontsize=8, color=colour, va='top', zorder=4)
            ax.text(x0 + 0.038, y, it, fontsize=7.3, color=DARK, va='top', zorder=4, linespacing=1.3)
            y -= line_h
        ax.text(x0 + w / 2, y0 - 0.055, note, ha='center', va='top', fontsize=7.4,
                color=DARK, style='italic', linespacing=1.4, zorder=4)

    fig.savefig(f'{OUT}/fig_scope_boundary.png', bbox_inches='tight'); plt.close(fig)


FIGURES = [fig_thesis_map, fig_literature_matrix, fig_scope_boundary]

if __name__ == '__main__':
    for f in FIGURES:
        f(); print('ok', f.__name__)
