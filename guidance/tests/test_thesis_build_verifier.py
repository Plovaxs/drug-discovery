"""Tests for the thesis build's own output verifier (guidance/thesis/build_thesis.py).

The verifier guards three faults that actually shipped in a built PDF and were caught only by reading
the rendered output by hand:

  1. A table caption whose label touched its text ("Table E.1Test-set ..."), because table captions used
     a tab separator without ever defining a tab stop wide enough for the label to advance past.
  2. A List-of-Tables page number that pointed at a prose cross-reference instead of the caption, so
     Table C.1 was listed on page 40 when its caption is on page 169.
  3. A stray CJK character typed by accident into a table note, which rendered into the PDF unnoticed.

A guard that is never exercised is not known to work, so each test here feeds the verifier a synthetic
page that contains the known fault and asserts that it fires, plus a clean counterpart that must not.
These run on fabricated page text, never on the real 230-page PDF, so the suite stays fast and does not
depend on a build having been run.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thesis'))

SKIP_EXIT = 2
try:
    import build_thesis as bt  # noqa: E402
except ImportError as e:
    # build_thesis needs python-docx, which lives in the thesis-building interpreter rather than the
    # project's conda environment. The runner executes every suite with the same interpreter, so this
    # suite has to announce itself as skipped rather than fail there; it is still exercised whenever
    # the thesis is built, and by running this file with the interpreter that has python-docx.
    print(f'SKIP: build_thesis not importable here ({e})')
    sys.exit(SKIP_EXIT)


class CaptionSpacingGuard(unittest.TestCase):
    """Fault 1: label touching caption text."""

    def _run(self, page_text, tabs):
        orig = bt.pdf_pages
        bt.pdf_pages = lambda _pdf: [page_text]
        try:
            return bt.verify_output('ignored.pdf', tabs, [])
        finally:
            bt.pdf_pages = orig

    def test_touching_label_is_reported(self):
        page = 'Table E.1Test-set predictive quality of every checkpoint\n'
        probs = self._run(page, [('E.1', 'Test-set predictive quality of every checkpoint')])
        self.assertTrue(any('touches caption text' in p for p in probs),
                        f'expected a spacing fault, got {probs}')

    def test_properly_spaced_label_is_accepted(self):
        page = 'Table E.1     Test-set predictive quality of every checkpoint\n'
        probs = self._run(page, [('E.1', 'Test-set predictive quality of every checkpoint')])
        self.assertEqual([p for p in probs if 'touches caption' in p], [])

    def test_prose_cross_reference_at_line_start_is_not_a_spacing_fault(self):
        """The regression that made an earlier version of this guard unusable: prose wraps so that a
        cross-reference begins a line ("Figure 5.1). The 24 comparisons ..."), and a position-based
        check reads the ')' as a missing space."""
        page = ('Figure 5.1). The 24 comparisons were corrected together.\n'
                'Figure 5.1     Track A, full tier. (A) Mean shift of the score.\n')
        orig = bt.pdf_pages
        bt.pdf_pages = lambda _pdf: [page]
        try:
            probs = bt.verify_output('ignored.pdf', [], [('5.1', 'Track A, full tier. (A) Mean shift')])
        finally:
            bt.pdf_pages = orig
        self.assertEqual([p for p in probs if 'touches caption' in p], [])


class ListPageGuard(unittest.TestCase):
    """Fault 2: a list entry pointing at the wrong page."""

    def _pages(self, listed_page):
        front = f'LIST OF TABLES\nC.1   Track A full tier: all comparisons{"." * 40}{listed_page}\n'
        ch1 = 'CHAPTER I\n'
        filler = 'body text mentioning Table C.1 in a sentence\n'
        caption = 'Table C.1     Track A full tier: all comparisons\n'
        return [front, ch1, filler, caption]  # caption sits on body page 3

    def _run(self, pages):
        orig = bt.pdf_pages
        bt.pdf_pages = lambda _pdf: pages
        try:
            return bt.verify_output('ignored.pdf', [], [])
        finally:
            bt.pdf_pages = orig

    def test_wrong_page_is_reported(self):
        probs = self._run(self._pages(40))
        self.assertTrue(any('listed on p.40' in p and 'p.3' in p for p in probs),
                        f'expected a page mismatch, got {probs}')

    def test_correct_page_is_accepted(self):
        probs = self._run(self._pages(3))
        self.assertEqual([p for p in probs if 'listed on p.' in p], [])


class CharsetGuard(unittest.TestCase):
    """Fault 3: a character that cannot have been intended."""

    def test_stray_cjk_character_is_reported(self):
        probs = bt._verify_charset(['the quantity that决 determined selection'])
        self.assertTrue(any('U+51B3' in p for p in probs), f'expected a charset fault, got {probs}')

    def test_scientific_symbols_are_accepted(self):
        page = 'λ = 1.0 gave −8.068 kcal/mol, p = 1.9 × 10⁻⁶, π-stacking, ΔR² ≈ 0.24, 10 µm, x ∈ S'
        self.assertEqual(bt._verify_charset([page]), [])


class MarkupBalanceGuard(unittest.TestCase):
    """Fault 4: unbalanced table markup, which the parser tolerates silently.

    An orphaned `@endtab` was introduced by hand while editing an appendix. It did not crash the
    parser and did not fail the build; it was found by counting tags. Since the failure mode is a
    document that builds successfully while quietly missing content, the check runs before parsing.
    """

    def _write(self, text):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'chapter.txt')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(text)
        return p

    def test_orphaned_endtab_is_rejected(self):
        p = self._write('@tab a | caption\n| k | v |\n@endtab\n\nprose\n@endtab\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_markup_balance([p])
        self.assertIn('no matching', str(cm.exception))

    def test_unclosed_table_is_rejected(self):
        p = self._write('@tab a | caption\n| k | v |\n\nprose with no @endtab\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_markup_balance([p])
        self.assertIn('never closed', str(cm.exception))

    def test_nested_table_open_is_rejected(self):
        p = self._write('@tab a | caption\n| k | v |\n@tab b | second\n| k | v |\n@endtab\n@endtab\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_markup_balance([p])
        self.assertIn('still open', str(cm.exception))

    def test_balanced_markup_is_accepted(self):
        p = self._write('@tab a | caption\n| k | v |\n@endtab\n\nprose\n\n@tab b | second\n| k | v |\n@endtab\n')
        bt.check_markup_balance([p])  # must not raise


class NumberSourceCheck(unittest.TestCase):
    """The quoted-number checker must agree with the live chapters, and must be strong enough to catch
    a single corrupted table cell.

    Its first version was not: it asked only whether a value occurred somewhere in the chapter, so
    corrupting one row of a table passed because the same figure also appeared in the prose nearby.
    The row-context test below pins the stronger behaviour."""

    def test_live_chapters_agree_with_their_sources(self):
        import check_numbers_against_sources as chk
        checked, problems = chk.check()
        real = [p for p in problems if 'cannot read' not in p]
        self.assertEqual(real, [], f'{len(real)} quoted value(s) disagree with their source')
        self.assertGreater(checked, 0, 'no claims were checked at all')

    def test_every_claim_can_actually_fail(self):
        """Mutation test: perturb each source value and require its claim to complain.

        A claim with no row context degenerates into "does this number appear anywhere in a 100 kB
        chapter", which almost anything satisfies by accident. Two entries were exactly that until this
        test found them. Enforcing non-vacuousness here stops a weak entry being added later and
        reported as coverage it does not provide."""
        import copy
        import check_numbers_against_sources as chk
        original_load, vacuous = chk.load, []
        try:
            for entry in chk.CLAIMS:
                label, src, path = entry[0], entry[1], entry[2]

                def mutated(rel, _s=src, _p=path):
                    data = copy.deepcopy(original_load(rel))
                    if rel == _s:
                        cur, parts = data, _p.split('.')
                        for part in parts[:-1]:
                            cur = cur[part]
                        val = cur[parts[-1]]
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            cur[parts[-1]] = val * 2 + 7
                    return data

                chk.load = mutated
                _, problems = chk.check()
                if not any(label in p for p in problems):
                    vacuous.append(label)
        finally:
            chk.load = original_load
        self.assertEqual(vacuous, [], f'{len(vacuous)} claim(s) pass no matter what the source says')

    def test_row_context_catches_a_single_corrupted_cell(self):
        """Simulate the exact failure the first version missed, without touching the real chapters."""
        import check_numbers_against_sources as chk
        raw = ('| Pocket-only (38 descriptors) | 38 | ridge | 0.9999 |\n'
               'prose that still mentions 0.2889 elsewhere\n')
        want, context = '0.2889', '| Pocket-only (38'
        hit = any(context in ln and want in ln for ln in raw.split('\n'))
        self.assertFalse(hit, 'row-context matching should not accept a corrupted cell')
        # and the weak form, which the first version used, would have wrongly accepted it
        self.assertIn(want, raw, 'the weak whole-chapter form would have passed -- that is the bug')


if __name__ == '__main__':
    # Report in this project's convention: one "PASS <name>" line per test on stdout, then a total.
    # guidance/run_tests.py counts those lines; a suite that only emits unittest's own stderr output is
    # recorded as "0 test(s)", which looks indistinguishable from a suite that silently ran nothing.
    cases = [CaptionSpacingGuard, ListPageGuard, CharsetGuard, MarkupBalanceGuard, NumberSourceCheck]
    loader = unittest.TestLoader()
    n = 0
    for cls in cases:
        for name in loader.getTestCaseNames(cls):
            res = unittest.TestResult()
            cls(name).run(res)
            if res.failures or res.errors:
                for _t, tb in res.failures + res.errors:
                    print(f'FAIL {cls.__name__}.{name}\n{tb}')
                raise SystemExit(1)
            print('PASS', f'{cls.__name__}.{name}')
            n += 1
    print(f'{n} tests passed')
