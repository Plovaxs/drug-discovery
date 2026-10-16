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

THESIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thesis')
sys.path.insert(0, THESIS)

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


class SectionRefGuard(unittest.TestCase):
    """Fault 8: prose pointing at a section number the document does not contain.

    Section numbers are positional. Inserting one `@sec` renumbers every later section in that
    chapter, and every "Section IV.9" written elsewhere silently starts pointing somewhere else.
    Nothing about that is visible in the source, in the build, or in the rendered PDF: the
    reference still looks perfectly well formed. On its first run this guard found three broken
    references that had survived every previous pass over the document, one of which had already
    been copied into a generated appendix.
    """

    def _write(self, text):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'chapter.txt')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(text)
        return p

    CH = ('@ch IV | PREDICTOR QUALITY\n@sec First\n@sub One\n@sub Two\n@sec Second\n'
          '@appendix A | REPRODUCIBILITY\n@sec Only\n')

    def test_numbering_matches_the_renderer(self):
        p = self._write(self.CH)
        self.assertEqual(bt.section_numbers([p]),
                         {'IV.1', 'IV.1.1', 'IV.1.2', 'IV.2', 'A.1'})

    def test_existing_reference_is_accepted(self):
        p = self._write(self.CH + 'prose citing Section IV.1.2 and Appendix A.1.\n')
        bt.check_section_refs([p])  # must not raise

    def test_missing_subsection_is_rejected(self):
        p = self._write(self.CH + 'prose citing Section IV.2.1, which does not exist.\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_section_refs([p])
        self.assertIn('IV.2.1', str(cm.exception))

    def test_chapter_form_is_also_checked(self):
        p = self._write(self.CH + 'prose citing Chapter IV.9.\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_section_refs([p])
        self.assertIn('IV.9', str(cm.exception))

    def test_bare_chapter_reference_is_not_flagged(self):
        """"Chapter IV" with no section number cannot go stale, so it is not checked."""
        p = self._write(self.CH + 'prose citing Chapter IV and Appendix A in general.\n')
        bt.check_section_refs([p])  # must not raise

    def test_comment_lines_are_ignored(self):
        p = self._write('# generated from Section IV.9.9\n' + self.CH)
        bt.check_section_refs([p])  # must not raise

    def test_live_chapters_have_no_broken_section_references(self):
        import glob
        files = sorted(glob.glob(os.path.join(THESIS, 'chapters', '*.txt')))
        if not files:
            self.skipTest('chapter sources not present')
        bt.check_section_refs(files)


class ManifestAndDuplicateHeadingGuard(unittest.TestCase):
    """Three further guards added in direct response to the same real incident as
    OrphanFragmentGuard (an external reviewer finding a duplicated appendix subsection,
    "E.7.1", in the committed PDF): an explicit chapter-file manifest instead of a bare glob,
    a source-level check that no two @sec/@sub titles repeat within one chapter/appendix, and a
    render-level check of the same thing against the actual rendered PDF text, as a layer that
    does not depend on the source-level parser having seen the problem correctly."""

    def test_manifest_catches_a_file_present_but_not_listed(self):
        import tempfile
        d = tempfile.mkdtemp()
        open(os.path.join(d, 'listed.txt'), 'w').close()
        open(os.path.join(d, 'stray.txt'), 'w').close()
        files = [os.path.join(d, 'listed.txt')]
        old = bt.CHAPTER_FILES
        bt.CHAPTER_FILES = ['listed.txt']
        try:
            with self.assertRaises(SystemExit) as cm:
                bt.check_manifest_matches_directory([os.path.join(d, 'stray.txt'),
                                                     os.path.join(d, 'listed.txt')])
            self.assertIn('stray.txt', str(cm.exception))
        finally:
            bt.CHAPTER_FILES = old

    def test_manifest_catches_a_listed_file_that_no_longer_exists(self):
        old = bt.CHAPTER_FILES
        bt.CHAPTER_FILES = ['this_file_does_not_exist_anywhere.txt']
        try:
            with self.assertRaises(SystemExit) as cm:
                bt.check_manifest_matches_directory([])
            self.assertIn('this_file_does_not_exist_anywhere.txt', str(cm.exception))
        finally:
            bt.CHAPTER_FILES = old

    def test_replace_exactly_once_applies_a_unique_match(self):
        self.assertEqual(bt.replace_exactly_once('a b c', 'b', 'X'), 'a X c')

    def test_replace_exactly_once_rejects_zero_matches(self):
        """The exact shape of the real bug: a fix that silently did nothing because its target
        string no longer matched, believed to have been applied, undetected for a full session."""
        with self.assertRaises(ValueError):
            bt.replace_exactly_once('a b c', 'ZZZ', 'X')

    def test_replace_exactly_once_rejects_multiple_matches(self):
        with self.assertRaises(ValueError):
            bt.replace_exactly_once('a b a', 'a', 'X')

    def test_duplicate_sub_title_within_one_appendix_is_rejected(self):
        blocks = [('appendix', 'E', 'TITLE'), ('sec', 'A section'),
                  ('sub', 'Method'), ('sub', 'Method')]
        with self.assertRaises(SystemExit) as cm:
            bt.check_no_duplicate_headings(blocks)
        self.assertIn('Method', str(cm.exception))

    def test_same_title_as_sec_and_sub_is_not_flagged(self):
        """The real, legitimate pattern this guard must not break: Chapter VI has both a top-level
        '@sec Design' (its own experiment) and, nested elsewhere, a '@sub Design' (the positive
        control's own sub-design) -- same word, different level, not a duplication bug."""
        blocks = [('chapter', 'VI', 'TITLE'), ('sec', 'Design'), ('sec', 'Results'),
                  ('sub', 'Design'), ('sub', 'Result')]
        bt.check_no_duplicate_headings(blocks)  # must not raise

    def test_same_title_in_different_chapters_is_not_flagged(self):
        blocks = [('chapter', 'IV', 'A'), ('sec', 'Method'),
                  ('chapter', 'V', 'B'), ('sec', 'Method')]
        bt.check_no_duplicate_headings(blocks)  # must not raise

    def test_pdf_level_duplicate_heading_detector_catches_a_synthetic_duplicate(self):
        """Unit-level test of the render-level guard, isolated from a full build: feeds it
        fabricated page text standing in for pdftotext output, so the detection logic is checked
        without needing a rendered PDF on disk."""
        toc = [(1, 'IV.1', 'A Section Title That Is Long Enough To Match', ('sec', 'IV.1'))]
        pages_clean = ['IV.1   A Section Title That Is Long Enough To Match\nsome body text\n']
        self.assertEqual(bt._verify_no_duplicate_headings_in_pdf(pages_clean, toc), [])
        pages_dup = ['IV.1   A Section Title That Is Long Enough To Match\n'
                     'more text\n'
                     'IV.1   A Section Title That Is Long Enough To Match\n']
        problems = bt._verify_no_duplicate_headings_in_pdf(pages_dup, toc)
        self.assertEqual(len(problems), 1)
        self.assertIn('IV.1', problems[0])

    def test_pdf_level_detector_does_not_flag_an_indented_table_of_contents_entry(self):
        """The false-positive this check produced on its first run against the real thesis: this
        document's Table of Contents has no dot leader before its page number (the assumption the
        first version of this guard relied on), so an indented TOC line for the same heading must
        not be counted as a second occurrence of the body heading."""
        toc = [(1, 'IV.1', 'A Section Title That Is Long Enough To Match', ('sec', 'IV.1'))]
        pages = ['    IV.1   A Section Title That Is Long Enough To Match      42\n'  # TOC, indented
                'IV.1   A Section Title That Is Long Enough To Match\nbody text\n']  # real heading
        self.assertEqual(bt._verify_no_duplicate_headings_in_pdf(pages, toc), [])

    def test_live_chapters_have_no_cross_chapter_manifest_drift(self):
        import glob
        files = sorted(glob.glob(os.path.join(THESIS, 'chapters', '*.txt')))
        if not files:
            self.skipTest('chapter sources not present')
        bt.check_manifest_matches_directory(files)


class DuplicateBodyContentWarning(unittest.TestCase):
    """A fourth guard suggested by the same external review: duplicate-heading checks key on
    titles by construction and cannot see a duplicated BODY sitting under two different titles.
    `warn_on_duplicate_body_content` is deliberately a warning, not a build failure -- two
    sections can legitimately share a long passage on purpose, and a hard failure on every such
    case would train whoever maintains this build to silence the check rather than read it."""

    def _blocks(self, sections):
        """sections: list of (label, kind, title, body) -> a flat block list like parse_sources
        produces, with one 'chapter' block introducing each distinct label."""
        out, seen_labels = [], set()
        for label, kind, title, body in sections:
            if label not in seen_labels:
                out.append(('chapter', label, 'TITLE')); seen_labels.add(label)
            out.append((kind, title))
            out.append(('para', body))
        return out

    def test_identical_long_body_under_different_titles_is_warned(self):
        body = 'x' * 250
        blocks = self._blocks([('IV', 'sub', 'First Title', body),
                               ('IV', 'sub', 'Second Title', body)])
        warned = bt.warn_on_duplicate_body_content(blocks)
        self.assertEqual(len(warned), 1)

    def test_short_identical_body_is_not_warned(self):
        body = 'Not applicable.'
        blocks = self._blocks([('IV', 'sub', 'First Title', body),
                               ('IV', 'sub', 'Second Title', body)])
        self.assertEqual(bt.warn_on_duplicate_body_content(blocks), [])

    def test_different_long_bodies_are_not_warned(self):
        blocks = self._blocks([('IV', 'sub', 'First Title', 'a' * 250),
                               ('IV', 'sub', 'Second Title', 'b' * 250)])
        self.assertEqual(bt.warn_on_duplicate_body_content(blocks), [])

    def test_identical_body_across_chapters_is_still_warned(self):
        """Unlike the title guard (scoped per chapter on purpose), a copy-pasted body is just as
        much a duplication risk whether it lands in the same chapter or a different one."""
        body = 'y' * 250
        blocks = self._blocks([('IV', 'sub', 'First Title', body),
                               ('VI', 'sub', 'Unrelated Title', body)])
        self.assertEqual(len(bt.warn_on_duplicate_body_content(blocks)), 1)

    def test_live_chapters_produce_no_duplicate_body_warnings(self):
        import glob
        files = sorted(glob.glob(os.path.join(THESIS, 'chapters', '*.txt')))
        if not files:
            self.skipTest('chapter sources not present')
        # parse_sources() re-runs every other guard too (balance, orphan-fragment, etc.), which is
        # fine: it is cheap, and exercising the real pipeline end to end is more representative
        # than re-implementing a slimmer parse just for this one check.
        blocks, _ = bt.parse_sources()
        warned = bt.warn_on_duplicate_body_content(blocks)
        self.assertEqual(warned, [], f'unexpected duplicate-body warning(s) on the live document: '
                                     f'{warned}')


class OrphanFragmentGuard(unittest.TestCase):
    """Fault 9: a fragment file under chapters/ with no @ch/@appendix of its own, silently
    absorbed into whichever chapter precedes it in sort order.

    This is not hypothetical. `make_chembl_expansion.py` writes a `@sub`-only fragment meant to
    be copied once into `11_appendices.txt` and then discarded; the fragment was left behind in
    `chapters/` after the copy. The build's glob picked it up a second time, and because it opened
    with `@sub` and no heading of its own, its content silently reattached to the end of whatever
    appendix the previous file in sort order had left open -- a full, legible second copy of
    "Was CD38 a pattern or a one-off? An expanded sample", numbered "E.7.1", stitched onto the
    tail of Appendix E. Every other guard in this file passed: the markup was balanced, no
    directive was buried in prose, and every section reference pointed at a section that existed
    (E.7.1 itself existed, it just should not have). An external reviewer reading the finished PDF
    found it; nothing in this build did until this guard was written in response.
    """

    def _write(self, text):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'chapter.txt')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(text)
        return p

    def test_orphan_fragment_is_rejected(self):
        p = self._write('@sub A fragment with no heading of its own\n\nsome prose.\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_no_orphan_fragments([p])
        self.assertIn(os.path.basename(p), str(cm.exception))

    def test_chapter_with_its_own_heading_is_accepted(self):
        p = self._write('@ch IV | PREDICTOR QUALITY\n\n@sec A section\n\nprose.\n')
        bt.check_no_orphan_fragments([p])  # must not raise

    def test_appendix_with_its_own_heading_is_accepted(self):
        p = self._write('@appendix F | COMPLETE EFFECT-SIZE REGISTER\n\n@sec A section\n\nprose.\n')
        bt.check_no_orphan_fragments([p])  # must not raise

    def test_abstract_file_is_accepted(self):
        p = self._write('@abstract\nprose.\n@endabstract\n')
        bt.check_no_orphan_fragments([p])  # must not raise

    def test_comment_lines_before_the_heading_do_not_confuse_the_check(self):
        p = self._write('# GENERATED FILE\n# Source: somewhere\n@appendix F | TITLE\n\nprose.\n')
        bt.check_no_orphan_fragments([p])  # must not raise

    def test_live_chapters_have_no_orphan_fragments(self):
        import glob
        files = sorted(glob.glob(os.path.join(THESIS, 'chapters', '*.txt')))
        if not files:
            self.skipTest('chapter sources not present')
        bt.check_no_orphan_fragments(files)


class GeneratedAppendix(unittest.TestCase):
    """Fault 7: a generated chapter file drifting behind the data it is generated from.

    Appendix F restates 71 comparisons that exist in machine-readable form, and is written by
    `make_effect_size_appendix.py` rather than by hand. That protects it from transcription error
    but introduces a different one: if a result file is updated and the generator is not re-run,
    the appendix keeps publishing the old numbers and nothing complains, because the file on disk
    is perfectly valid markup. Regenerating it here and comparing byte for byte closes that gap.
    """

    def test_chembl_expansion_is_not_stale(self):
        """Checks staleness against the real target, not a file that no longer exists.

        Until the orphan-fragment bug (see OrphanFragmentGuard) was found and fixed, the
        generator's committed output lived at `chapters/11y_chembl_expansion.txt`, a second,
        independent copy of content that was ALSO hand-spliced into `11_appendices.txt` -- the
        duplication the external reviewer found. The fix deleted that second copy and made
        `11_appendices.txt` the sole, authoritative home of this content; the generator now writes
        to `fragments/` (outside the chapter glob) for comparison only. This test's old form
        checked the generator's output against the path that was JUST DELETED as part of that
        fix, so it was silently skipping (via the `not os.path.exists(committed)` branch) and
        reporting as a pass with nothing actually checked -- a second instance of the same root
        problem (a check whose target quietly stopped existing) one guard below Issue 9 itself.
        """
        import subprocess
        import tempfile
        gen = os.path.join(THESIS, 'make_chembl_expansion.py')
        committed = os.path.join(THESIS, 'chapters', '11_appendices.txt')
        if not (os.path.exists(gen) and os.path.exists(committed)):
            self.skipTest('generator or appendix file absent')
        root = os.path.dirname(THESIS.rstrip(os.sep))
        root = os.path.dirname(root) if os.path.basename(root) == 'guidance' else root
        with tempfile.TemporaryDirectory() as tmp:
            fresh = os.path.join(tmp, 'fresh.txt')
            env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
            proc = subprocess.run([sys.executable, gen, '--out', fresh], cwd=root, env=env,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                out = (proc.stdout + proc.stderr)
                if 'ModuleNotFoundError' in out or 'FileNotFoundError' in out:
                    self.skipTest('generator cannot run in this interpreter / data layout')
                self.fail(f'generator exited {proc.returncode}:\n{out[-800:]}')
            fresh_body = '\n'.join(l for l in open(fresh, encoding='utf-8').read().split('\n')
                                   if not l.startswith('#')).strip()
        appendix = open(committed, encoding='utf-8').read()
        self.assertIn(fresh_body, appendix,
                     'the ChEMBL expansion content embedded in 11_appendices.txt no longer '
                     'matches what the generator produces: regenerate with PYTHONPATH=. python '
                     'guidance/thesis/make_chembl_expansion.py and re-splice by hand, or update '
                     'the generator to match an intentional manual edit')

    def test_effect_size_appendix_is_not_stale(self):
        import subprocess
        import tempfile
        gen = os.path.join(THESIS, 'make_effect_size_appendix.py')
        committed = os.path.join(THESIS, 'chapters', '11z_effect_sizes.txt')
        if not (os.path.exists(gen) and os.path.exists(committed)):
            self.skipTest('generator or generated appendix absent')
        root = os.path.dirname(THESIS.rstrip(os.sep))
        root = os.path.dirname(root) if os.path.basename(root) == 'guidance' else root
        with tempfile.TemporaryDirectory() as tmp:
            fresh = os.path.join(tmp, 'fresh.txt')
            env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
            proc = subprocess.run([sys.executable, gen, '--out', fresh], cwd=root, env=env,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                out = (proc.stdout + proc.stderr)
                if 'ModuleNotFoundError' in out or 'FileNotFoundError' in out:
                    self.skipTest('generator cannot run in this interpreter / data layout')
                self.fail(f'generator exited {proc.returncode}:\n{out[-800:]}')
            a = open(committed, encoding='utf-8').read()
            b = open(fresh, encoding='utf-8').read()
        self.assertEqual(a, b, 'Appendix F is stale: regenerate it with '
                               'PYTHONPATH=. python guidance/thesis/make_effect_size_appendix.py')


class TableDerivedFigures(unittest.TestCase):
    """Fault 6: a chapter table and the figure drawn from it drifting apart.

    The thesis map of Chapter I and the literature matrix of Chapter II are generated by parsing
    `@tab objmap` and `@tab lit` out of the chapter sources. Those parsers reconcile every table
    row against the figure's own layout and refuse to draw anything they cannot match, but that
    check only runs when somebody regenerates the figures. Running it here means an edit to either
    table is caught on the next test run instead of silently leaving a stale picture in the PDF.
    """

    def _layouts(self):
        """Import the layout maps without importing matplotlib, which this interpreter lacks."""
        import ast
        src = os.path.join(THESIS, 'make_figures_chapters.py')
        tree = ast.parse(open(src, encoding='utf-8').read())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], 'id', None)
                if name in ('THESIS_MAP', 'LIT_WORKS'):
                    out[name] = ast.literal_eval(node.value)
        return out

    def test_objmap_table_and_thesis_map_figure_agree(self):
        import thesis_tables as tt
        layout = self._layouts()['THESIS_MAP']
        _h, rows = tt.read_table('01_introduction.txt', 'objmap')
        pairs = tt.match_rows(rows, layout, col=0, what='thesis map')
        self.assertEqual(len(pairs), len(rows))
        self.assertEqual(len({k for k, _ in pairs}), len(layout),
                         'every layout entry must match exactly one row')

    def test_lit_table_and_matrix_figure_agree(self):
        import thesis_tables as tt
        layout = self._layouts()['LIT_WORKS']
        _h, rows = tt.read_table('02_literature.txt', 'lit')
        pairs = tt.match_rows(rows, layout, col=0, what='literature matrix')
        self.assertEqual(len(pairs), len(rows))

    def test_every_literature_cell_classifies(self):
        """A cell written in an unrecognised style must stop the build, not be guessed at."""
        import thesis_tables as tt
        header, rows = tt.read_table('02_literature.txt', 'lit')
        for row in rows:
            for cell in row[1:]:
                self.assertIn(tt.classify(cell), ('yes', 'no', 'unknown'))

    def test_unrecognised_cell_is_rejected(self):
        import thesis_tables as tt
        with self.assertRaises(SystemExit):
            tt.classify('Proposed as a generation technique, not framed as a control')

    def test_unmatched_row_is_rejected(self):
        import thesis_tables as tt
        with self.assertRaises(SystemExit) as cm:
            tt.match_rows([['something the layout never mentions']],
                          {'a': ('a probe that is absent',)}, what='probe')
        self.assertIn('expected 1', str(cm.exception))

    def test_unused_layout_entry_is_rejected(self):
        import thesis_tables as tt
        with self.assertRaises(SystemExit) as cm:
            tt.match_rows([['row mentioning alpha']],
                          {'a': ('alpha',), 'b': ('beta',)}, what='probe')
        self.assertIn('never matched', str(cm.exception))


class InlineDirectiveGuard(unittest.TestCase):
    """Fault 5: a directive appended to the end of a line of prose.

    The parser recognises directives only with ``startswith``, so one that ends up inside a
    paragraph is rendered as body text and the heading, figure or table it was meant to produce
    silently never appears. This happened to a whole `@sub` heading in Chapter V while an existing
    paragraph was being extended by hand, and the build reported success. Like unbalanced table
    markup, the failure mode is a document that builds cleanly while missing content, so the check
    runs before parsing.
    """

    def _write(self, text):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'chapter.txt')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(text)
        return p

    def test_sub_buried_in_prose_is_rejected(self):
        p = self._write('Some prose that runs on. @sub The heading that vanishes\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_inline_directives([p])
        self.assertIn('@sub', str(cm.exception))

    def test_fig_buried_in_prose_is_rejected(self):
        p = self._write('prose\n\nmore prose. @fig k | path.png | 5.0 | caption\n')
        with self.assertRaises(SystemExit) as cm:
            bt.check_inline_directives([p])
        self.assertIn('@fig', str(cm.exception))

    def test_line_initial_directives_are_accepted(self):
        p = self._write('@sec A section\n\nprose\n\n@sub A subheading\n\n'
                        '@fig k | path.png | 5.0 | caption\n\n@tab t | caption\n| k | v |\n@endtab\n')
        bt.check_inline_directives([p])  # must not raise

    def test_directive_payload_on_its_own_line_is_not_a_false_positive(self):
        """A real directive's own line may legitimately contain '@' text after the marker."""
        p = self._write('@fig k | path.png | 5.0 | A caption mentioning @sub as a literal word\n')
        bt.check_inline_directives([p])  # must not raise

    def test_email_like_at_token_in_prose_is_not_flagged(self):
        p = self._write('prose citing [@rucker2007] and an @unknown token that is not a directive\n')
        bt.check_inline_directives([p])  # must not raise


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

                def step(cur, part):
                    """Walk one path element the way check_numbers_against_sources.jpath does.

                    Claim paths address list elements by integer index (``ci95.0``) and a sole dict
                    value by ``*``, so a walker that only does ``cur[part]`` raises TypeError on the
                    first interval claim and aborts the whole mutation sweep. It did.
                    """
                    if part == '*':
                        return next(iter(cur.values()))
                    return cur[int(part)] if isinstance(cur, list) else cur[part]

                def mutated(rel, _s=src, _p=path):
                    data = copy.deepcopy(original_load(rel))
                    if rel == _s:
                        cur, parts = data, _p.split('.')
                        for part in parts[:-1]:
                            cur = step(cur, part)
                        last = parts[-1]
                        key = int(last) if isinstance(cur, list) else last
                        if last == '*':
                            key = next(iter(cur))
                        val = cur[key]
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            cur[key] = val * 2 + 7
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
    cases = [CaptionSpacingGuard, ListPageGuard, CharsetGuard, MarkupBalanceGuard,
             InlineDirectiveGuard, TableDerivedFigures, SectionRefGuard, GeneratedAppendix,
             OrphanFragmentGuard, ManifestAndDuplicateHeadingGuard, DuplicateBodyContentWarning,
             NumberSourceCheck]
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
