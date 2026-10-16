"""Reading the thesis's own tables back out of the chapter sources.

Two figures are drawn from tables rather than from result files: the thesis map of Chapter I and
the literature matrix of Chapter II. Both must say exactly what their table says, and neither has
a JSON file behind it that `check_numbers_against_sources.py` could pin. The protection instead is
that the figures parse the chapter source and refuse to draw anything they cannot reconcile with
it.

This module holds that parsing with no plotting dependency, for two reasons. The figure modules
need matplotlib, which the interpreter that runs the test suite does not have, while the suite
needs this logic, which is the part that can silently go wrong. Keeping them apart lets the
synchronisation between table and figure be tested on every run rather than only when someone
remembers to regenerate the figures.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = os.path.join(HERE, 'chapters')


def read_table(chapter, key, chapters_dir=None):
    """Return (header, rows) of the `@tab <key>` block in `chapters/<chapter>`.

    Rows are lists of cell strings with the leading and trailing pipe removed. A missing table is
    an error rather than an empty result, because a figure drawn from a table that is no longer
    there would otherwise be drawn from nothing at all.
    """
    path = os.path.join(chapters_dir or CHAPTERS, chapter)
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(f'@tab {key} |'):
            start = i
            break
    if start is None:
        raise SystemExit(f'{chapter}: no "@tab {key}" block found; a figure depends on it')
    rows = []
    for ln in lines[start + 1:]:
        if ln.startswith('@endtab'):
            break
        if ln.startswith('|'):
            rows.append([c.strip() for c in ln.strip().strip('|').split('|')])
    if len(rows) < 2:
        raise SystemExit(f'{chapter}: "@tab {key}" has fewer than two rows')
    return rows[0], rows[1:]


def match_rows(rows, layout, col=0, what='row'):
    """Pair each parsed table row with the layout entry whose probe substring it contains.

    Both directions are checked. An unmatched row means the chapter grew something the figure does
    not draw; an unused layout entry means the figure still draws something the chapter no longer
    says. Either one makes the figure wrong, so either one stops the build. An ambiguous row, which
    two probes both match, is also an error: silently taking the first would make the pairing
    depend on dictionary order.
    """
    out, used = [], set()
    for r in rows:
        hits = [k for k, (probe, *_rest) in layout.items() if probe in r[col]]
        if len(hits) != 1:
            raise SystemExit(f'{what}: {len(hits)} layout entries match "{r[col][:70]}..."; expected 1')
        out.append((hits[0], r))
        used.add(hits[0])
    missing = set(layout) - used
    if missing:
        raise SystemExit(f'{what}: layout entries never matched any table row: {sorted(missing)}')
    return out


def classify(cell):
    """Map a literature-table cell to one of three display states, strictly.

    An earlier version guessed: anything that did not begin with a negative marker was treated as
    "present". That silently painted one BoKDiff cell green although the cell says the feature was
    *not* framed that way, which would have overstated a prior work's claims in a figure about
    exactly which claims each prior work makes. Rather than add another prefix rule, this raises on
    any cell it does not recognise, so a future cell written in a new style stops the build instead
    of being quietly mis-coloured. The "unknown" state is kept distinct from "absent" on purpose:
    the chapter is careful to separate a feature a study lacks from one its reviewed material
    simply does not state, and collapsing the two would misrepresent the prior work.
    """
    low = cell.strip().lower()
    if low.startswith('not stated') or low.startswith('not the focus'):
        return 'unknown'
    if low.startswith('no') or low.startswith('n/a'):
        return 'no'
    if low.startswith('yes') or low.startswith('plip') or low.startswith('binana') \
            or low.startswith('latent priors') or low.startswith('contact map'):
        return 'yes'
    raise SystemExit('literature matrix: cell does not begin with a recognised marker '
                     f'(Yes / No / n/a / Not stated / Not the focus): {cell[:90]!r}')
