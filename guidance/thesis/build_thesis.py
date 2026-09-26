"""Builds the thesis (.docx + .pdf) in the President University Faculty of Computing format
("Thesis Format Writing Guide", Master of Informatics).

  ~/... python guidance/thesis/build_thesis.py            # uses the isolated venv that has python-docx
Sources: guidance/thesis/chapters/*.txt (mini-markup below), guidance/thesis/refs.py, guidance/thesis/figures/*.png,
         guidance/thesis_figures/*.png.  Two passes: pass 1 builds with placeholder page numbers, LibreOffice converts to PDF,
         page numbers of every heading/table/figure are read from the PDF, pass 2 rebuilds with real numbers.

Chapter mini-markup (one directive per line, paragraphs separated by blank lines):
  @ch I | INTRODUCTION            chapter heading            @appendix A | TITLE      appendix heading
  @sec Title / @sub Title         numbered section / subsection (numbers automatic)
  @bul                            following "- " lines (until blank line) form a list
  @fig key | path | width_in | caption
  @tab key | caption             then "| a | b |" rows (first row = header), then "@endtab"; optional "@note text" before @endtab
  @eq expression | key           numbered equation            @pb   page break
  Inline: **bold**, *italic*, ^{sup}, _{sub}, [[highlighted placeholder]], [@refkey] or [@a; @b] citations,
          {fig:key} {tab:key} {eq:key} cross-references.
"""
import glob
import os
import re
import subprocess
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refs import REFS, ordered_keys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DOCX = os.path.join(HERE, 'Thesis_Coupled_Guidance_SBDD.docx')
FONT = 'Times New Roman'
TEXT_W = 8.27 - 1.5 - 1.0          # A4 width minus margins (in)

META = dict(
    title_lines=['COUPLED AFFINITY-SYNTHESIZABILITY GUIDANCE FOR', 'TARGET-CONDITIONAL MOLECULAR DIFFUSION IN', 'STRUCTURE-BASED DRUG DESIGN'],
    title_inline='Coupled Affinity-Synthesizability Guidance for Target-Conditional Molecular Diffusion in Structure-Based Drug Design',
    name='[[STUDENT NAME]]', sid='[[STUDENT IDENTIFICATION NUMBER]]', when='[[MONTH, YEAR OF THE DEFENSE]]',
    degree='[[MAGISTER — academic degree as listed in the student handbook]]',
    advisor='[[Your Advisor Name]]',
)
ROMAN = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12}


# ------------------------------------------------------------------------------------------------------- parsing
def parse_sources():
    blocks, abstract, abbrevs = [], None, []
    files = sorted(glob.glob(os.path.join(HERE, 'chapters', '*.txt')))
    for f in files:
        lines = open(f, encoding='utf-8').read().split('\n')
        i = 0
        para = []

        def flush():
            nonlocal para
            if para:
                blocks.append(('para', ' '.join(s.strip() for s in para)))
                para = []
        while i < len(lines):
            ln = lines[i]
            st = ln.strip()
            if not st:
                flush(); i += 1; continue
            if st.startswith('#'):
                i += 1; continue
            if st.startswith('@abstract'):
                flush(); i += 1; buf = []
                while i < len(lines) and not lines[i].startswith('@endabstract'):
                    buf.append(lines[i]); i += 1
                abstract = '\n'.join(buf).strip(); i += 1; continue
            if st.startswith('@ch '):
                flush(); a, b = [x.strip() for x in st[4:].split('|', 1)]; blocks.append(('chapter', a, b)); i += 1; continue
            if st.startswith('@appendix '):
                flush(); a, b = [x.strip() for x in st[10:].split('|', 1)]; blocks.append(('appendix', a, b)); i += 1; continue
            if st.startswith('@sec '):
                flush(); blocks.append(('sec', st[5:].strip())); i += 1; continue
            if st.startswith('@sub '):
                flush(); blocks.append(('sub', st[5:].strip())); i += 1; continue
            if st.startswith('@pb'):
                flush(); blocks.append(('pb',)); i += 1; continue
            if st.startswith('@bul'):
                flush(); i += 1; items = []
                while i < len(lines) and lines[i].strip().startswith('- '):
                    items.append(lines[i].strip()[2:]); i += 1
                blocks.append(('bul', items)); continue
            if st.startswith('@fig '):
                flush(); parts = [x.strip() for x in st[5:].split('|', 3)]
                blocks.append(('fig', parts[0], parts[1], float(parts[2]), parts[3])); i += 1; continue
            if st.startswith('@eq '):
                flush(); a, b = [x.strip() for x in st[4:].rsplit('|', 1)]; blocks.append(('eq', a, b)); i += 1; continue
            if st.startswith('@tab '):
                flush(); key, cap = [x.strip() for x in st[5:].split('|', 1)]; i += 1; rows, note = [], ''
                while i < len(lines) and not lines[i].startswith('@endtab'):
                    s2 = lines[i].strip()
                    if s2.startswith('@note '):
                        note = s2[6:]
                    elif s2.startswith('|'):
                        rows.append([c.strip() for c in s2.strip('|').split('|')])
                    i += 1
                blocks.append(('tab', key, cap, rows, note)); i += 1; continue
            para.append(ln); i += 1
        flush()
    return blocks, abstract


def assign_numbers(blocks):
    """Numbers for sections/tables/figures/equations + cross-reference map + TOC entries."""
    xref, toc, tabs, figs = {}, [], [], []
    ch_arabic, cur, sec_n, sub_n, tab_n, fig_n, eq_n, label = 0, None, 0, 0, 0, 0, 0, None
    out = []
    for b in blocks:
        if b[0] == 'chapter':
            ch_arabic = ROMAN[b[1]]; label = b[1]; sec_n = sub_n = tab_n = fig_n = eq_n = 0
            toc.append((0, b[1] + '.', b[2], ('ch', b[1]))); out.append(b)
        elif b[0] == 'appendix':
            label = b[1]; ch_arabic = b[1]; sec_n = sub_n = tab_n = fig_n = eq_n = 0
            toc.append((0, 'APPENDIX ' + b[1], b[2], ('ap', b[1]))); out.append(b)
        elif b[0] == 'sec':
            sec_n += 1; sub_n = 0; num = f'{label}.{sec_n}'; toc.append((1, num, b[1], ('sec', num))); out.append(('sec', num, b[1]))
        elif b[0] == 'sub':
            sub_n += 1; num = f'{label}.{sec_n}.{sub_n}'; toc.append((2, num, b[1], ('sub', num))); out.append(('sub', num, b[1]))
        elif b[0] == 'tab':
            tab_n += 1; num = f'{ch_arabic}.{tab_n}'; xref['tab:' + b[1]] = f'Table {num}'; tabs.append((num, b[2])); out.append(('tab', num) + b[1:])
        elif b[0] == 'fig':
            fig_n += 1; num = f'{ch_arabic}.{fig_n}'; xref['fig:' + b[1]] = f'Figure {num}'; figs.append((num, b[4])); out.append(('fig', num) + b[1:])
        elif b[0] == 'eq':
            eq_n += 1; num = f'({ch_arabic}.{eq_n})'; xref['eq:' + b[2]] = num; out.append(('eq', num) + b[1:])
        else:
            out.append(b)
    return out, xref, toc, tabs, figs


# ------------------------------------------------------------------------------------------------------- docx helpers
def set_font(run, size=12, bold=None, italic=None, sup=False, sub=False):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if sup:
        run.font.superscript = True
    if sub:
        run.font.subscript = True


TOKEN = re.compile(r'(\*\*.+?\*\*|\*[^*\s][^*]*?\*|\^\{.+?\}|_\{.+?\}|\[\[.+?\]\])')


def add_runs(par, text, size=12, bold=False, italic=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            r = par.add_run(part[2:-2]); set_font(r, size, True, italic)
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            r = par.add_run(part[1:-1]); set_font(r, size, bold, not italic)
        elif part.startswith('^{'):
            r = par.add_run(part[2:-1]); set_font(r, size, bold, italic, sup=True)
        elif part.startswith('_{'):
            r = par.add_run(part[2:-1]); set_font(r, size, bold, italic, sub=True)
        elif part.startswith('[[') and part.endswith(']]'):
            r = par.add_run(part[2:-2]); set_font(r, size, bold, italic); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            r = par.add_run(part); set_font(r, size, bold, italic)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=2.0, first=None, left=None, before=0, after=0, keep_next=False, keep_lines=False):
    pf = par.paragraph_format
    par.alignment = align
    pf.line_spacing = spacing
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if left is not None:
        pf.left_indent = Inches(left)
    pf.widow_control = True
    pf.keep_with_next = keep_next
    pf.keep_together = keep_lines


def page_field(par, size=12):
    r = par.add_run(); set_font(r, size)
    for typ, txt in (('begin', None), (None, 'PAGE'), ('separate', None), (None, '1'), ('end', None)):
        if typ:
            e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), typ); r._r.append(e)
        elif txt == 'PAGE':
            e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = ' PAGE '; r._r.append(e)
        else:
            e = OxmlElement('w:t'); e.text = txt; r._r.append(e)


def setup_section(sec, top=1.25, bottom=1.25, fmt_=None, start=None, first_diff=False):
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin, sec.right_margin = Inches(1.5), Inches(1.0)
    sec.top_margin, sec.bottom_margin = Inches(top), Inches(bottom)
    sec.header_distance, sec.footer_distance = Inches(0.6), Inches(0.6)
    sec.different_first_page_header_footer = first_diff
    sp = sec._sectPr
    for e in sp.findall(qn('w:pgNumType')):
        sp.remove(e)
    if fmt_ or start is not None:
        e = OxmlElement('w:pgNumType')
        if fmt_:
            e.set(qn('w:fmt'), fmt_)
        if start is not None:
            e.set(qn('w:start'), str(start))
        sp.append(e)


def clear_hf(hf):
    hf.is_linked_to_previous = False
    for p in list(hf.paragraphs)[1:]:
        p._element.getparent().remove(p._element)
    hf.paragraphs[0].text = ''
    return hf.paragraphs[0]


def hf_page(hf, align):
    p = clear_hf(hf); p.alignment = align; page_field(p)


def empty_hf(sec):
    for hf in (sec.header, sec.footer, sec.first_page_header, sec.first_page_footer):
        clear_hf(hf)


# ------------------------------------------------------------------------------------------------------- document build
class Builder:
    def __init__(self, blocks, abstract, xref, toc, tabs, figs, pages, pages_scan_extra=()):
        self.blocks, self.abstract, self.xref, self.toc, self.tabs, self.figs = blocks, abstract, xref, toc, tabs, figs
        self.pages = pages or {}
        self.cited = set()
        self.order = ordered_keys()
        # numbers are assigned among CITED references only, alphabetically (template Sec. IV.3.1)
        found = set()
        def scan(x):
            if isinstance(x, str):
                for m in re.finditer(r'\[(@[^\]]+)\]', x):
                    found.update(k.strip().lstrip('@') for k in m.group(1).split(';'))
            elif isinstance(x, (list, tuple)):
                for y in x:
                    scan(y)
        scan(list(blocks)); scan(abstract); scan(pages_scan_extra)
        self.cite_order = [k for k in self.order if k in found]
        self.cite_no = {k: i + 1 for i, k in enumerate(self.cite_order)}
        self.doc = Document()
        st = self.doc.styles['Normal']
        st.font.name = FONT; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
        self.first_section_used = False

    # ---- text with citations / xrefs ----
    def resolve(self, text):
        def cite(m):
            keys = [k.strip().lstrip('@') for k in m.group(1).split(';')]
            nums = []
            for k in keys:
                if k not in self.cite_no:
                    raise KeyError(f'unknown reference key: {k}')
                self.cited.add(k); nums.append(self.cite_no[k])
            return ', '.join(f'[{n}]' for n in sorted(nums))
        text = re.sub(r'\[(@[^\]]+)\]', cite, text)
        text = re.sub(r'\{(fig|tab|eq):([^}]+)\}', lambda m: self.xref[m.group(1) + ':' + m.group(2)], text)
        return text

    def pg(self, key, roman=False):
        v = self.pages.get(key)
        return '' if v is None else str(v)

    # ---- paragraph primitives ----
    def para(self, text, **kw):
        p = self.doc.add_paragraph(); add_runs(p, self.resolve(text)); fmt(p, first=0.5, **kw); return p

    def centered(self, text, size=12, bold=False, before=0, after=0, spacing=1.0, italic=False):
        p = self.doc.add_paragraph(); add_runs(p, text, size, bold, italic); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, spacing, before=before, after=after); return p

    def blank(self, n=1):
        for _ in range(n):
            p = self.doc.add_paragraph(); fmt(p, spacing=1.0)

    def page_break(self):
        p = self.doc.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE); fmt(p, spacing=1.0)

    def new_section(self, **kw):
        if not self.first_section_used:
            sec = self.doc.sections[0]; self.first_section_used = True
        else:
            sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        setup_section(sec, **kw)
        return sec

    # ---- front matter ----
    def front(self):
        d = self.doc
        sec = self.new_section(top=1.0, bottom=1.0)
        empty_hf(sec)
        # title page
        self.blank(6)
        for i, t in enumerate(META['title_lines']):
            self.centered(t, 12, True, after=0)
        self.blank(2)
        self.centered('THESIS', 12, before=18, after=18)
        self.centered('Submitted as one of the requirements to obtain', 12)
        self.centered(META['degree'], 12, after=0)
        self.blank(3)
        self.centered('By:', 12, after=12)
        self.centered(META['name'], 12); self.centered(META['sid'], 12)
        self.blank(2)
        self.centered('FACULTY OF COMPUTER SCIENCE', 12); self.centered('MASTER OF INFORMATICS STUDY PROGRAM', 12)
        self.centered('PRESIDENT UNIVERSITY', 12); self.centered('CIKARANG', 12); self.centered(META['when'], 12)
        self.page_break()
        # copyright
        self.blank(14); self.centered('Copyright by'); self.centered(META['name']); self.centered('[[YEAR]]')
        self.page_break()
        # approval
        for t in META['title_lines']:
            self.centered(t, 12, True)
        self.blank(1); self.centered('By', before=12); self.centered(META['sid'], before=12); self.centered('Approved:', before=12, after=36)
        t = d.add_table(rows=2, cols=2)
        cells = [(META['advisor'], 'Thesis Advisor'), ('Ahmad Fadhil N, Ph.D.', 'Program Head of MIT')]
        for c, (n, r) in zip(t.rows[0].cells, cells):
            c.text = ''; p = c.paragraphs[0]; add_runs(p, '______________________________'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0)
            p2 = c.add_paragraph(); add_runs(p2, n); fmt(p2, WD_ALIGN_PARAGRAPH.LEFT, 1.0)
            p3 = c.add_paragraph(); add_runs(p3, r); fmt(p3, WD_ALIGN_PARAGRAPH.LEFT, 1.0)
        self.blank(3)
        self.centered('_________________________________', before=24); self.centered('Prof. Dr. Ir. Wiranto Herry Utomo, M.Kom.'); self.centered('Dean of Faculty of Computer Science')
        self.page_break()
        # originality
        self.centered('STATEMENT OF ORIGINALITY', 14, True, before=36, after=24)
        for line in ['In my capacity as an active student of President University and as the author of the thesis stated below:']:
            self.para_plain(line)
        for a, b in [('Name', META['name']), ('Student ID number', META['sid']), ('Study Program', 'Master of Informatics'), ('Faculty', 'Computer Science')]:
            p = d.add_paragraph(); add_runs(p, f'{a}\t: {b}'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(1.9))
        self.para_plain('I hereby declare that my thesis entitled “' + META['title_inline'] + '” is, to the best of my knowledge and belief, an original piece of work based on sound academic principles. If any plagiarism is detected in this thesis, I am willing to be personally responsible for the consequences of these acts of plagiarism, and will accept the sanctions against these acts in accordance with the rules and policies of President University.', before=12)
        self.para_plain('I also declare that this work, either in whole or in part, has not been submitted to another university to obtain a degree.')
        self.centered('Cikarang, [[DATE]]', before=24, after=48).alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.centered('( [[Full name & signature]] )', after=0).alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.page_break()
        # publication approval
        self.centered('SCIENTIFIC PUBLICATION APPROVAL FOR ACADEMIC INTEREST', 14, True, before=36, after=24)
        self.para_plain('As an academic community member of President University, I, the undersigned:')
        for a, b in [('Name', META['name']), ('Student ID number', META['sid']), ('Study program', 'Master of Informatics')]:
            p = d.add_paragraph(); add_runs(p, f'{a}\t: {b}'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0); p.paragraph_format.tab_stops.add_tab_stop(Inches(1.9))
        self.para_plain('for the purpose of development of science and technology, certify and approve to give President University a non-exclusive royalty-free right upon my final report with the title: “' + META['title_inline'] + '”.', before=12)
        self.para_plain('With this non-exclusive royalty-free right, President University is entitled to converse, to convert, to manage in a database, to maintain, and to publish my final report, with the obligation of President University to mention my name as the copyright owner of my final report.')
        self.para_plain('This statement I made in truth.')
        self.centered('Cikarang, [[DATE]]', before=24, after=48).alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.centered('( [[Full name & signature]] )').alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.page_break()
        # advisor approval
        self.centered('ADVISOR APPROVAL FOR PUBLICATION', 14, True, before=36, after=24)
        self.para_plain('As an academic community member of President University, I, the undersigned:')
        for a, b in [('Advisor Name', META['advisor']), ('Employee ID number', '[[EMPLOYEE ID]]'), ('Study program', 'Master of Informatics'), ('Faculty', 'Computer Science')]:
            p = d.add_paragraph(); add_runs(p, f'{a}\t: {b}'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0); p.paragraph_format.tab_stops.add_tab_stop(Inches(1.9))
        self.para_plain('declare that the following thesis:', before=12)
        for a, b in [('Title of thesis', META['title_inline']), ('Thesis author', META['name']), ('Student ID number', META['sid'])]:
            p = d.add_paragraph(); add_runs(p, f'{a}\t: {b}'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, left=1.9, first=-1.9); p.paragraph_format.tab_stops.add_tab_stop(Inches(1.9))
        self.para_plain('will be published in: journal / institution’s repository / proceeding / unpublished / [[other]] (choose one, underline that applies).', before=12)
        self.centered('Cikarang, [[DATE]]', before=24, after=48).alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.centered('( [[Advisor full name & signature]] )').alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self.page_break()
        # similarity index
        self.centered('SIMILARITY INDEX REPORT', 14, True, before=36, after=24)
        self.centered('[[Insert the similarity-check (Turnitin) report for the final version of this thesis here.]]', before=72)
        self.page_break()
        # abstract
        self.centered('ABSTRACT', 12, True, before=36, after=18)
        paras = [p.strip() for p in self.abstract.split('\n\n') if p.strip()]
        for pt in paras:
            if pt.lower().startswith('keywords'):
                p = d.add_paragraph(); add_runs(p, pt); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 2.0, before=6)
            else:
                self.para(pt)
        # ---- roman-numbered preliminary pages ----
        sec = self.new_section(top=1.25, bottom=1.25, fmt_='lowerRoman', start=2)
        for hf in (sec.header, sec.first_page_header, sec.first_page_footer):
            clear_hf(hf)
        hf_page(sec.footer, WD_ALIGN_PARAGRAPH.CENTER)
        self.centered('DEDICATION', 12, True, before=54, after=18)
        self.para('[[To be written by the author (optional page).]]')
        self.page_break()
        self.centered('ACKNOWLEDGMENTS', 12, True, before=54, after=18)
        self.para('[[The author wishes to express… (to be written by the author).]]')
        self.page_break()
        self.toc_pages()

    def para_plain(self, text, before=0, after=6):
        p = self.doc.add_paragraph(); add_runs(p, text); fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.5, before=before, after=after); return p

    # ---- tables of contents ----
    def toc_line(self, num, title, page, level, bold=False, gap=0):
        p = self.doc.add_paragraph()
        indent = [0.0, 0.35, 0.85][level]
        num_w = [0.5, 0.55, 0.75][level] if num else 0.0
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, left=indent + num_w, first=-num_w, before=gap, after=2)
        pf = p.paragraph_format
        pf.tab_stops.add_tab_stop(Inches(indent + num_w))
        pf.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        pf.right_indent = Inches(0.0)
        add_runs(p, (f'{num}\t' if num else '') + f'{title}\t{page}', 12, bold)
        return p

    def toc_pages(self):
        self.centered('TABLE OF CONTENTS', 12, True, before=0, after=18)
        p = self.doc.add_paragraph(); add_runs(p, 'Page'); fmt(p, WD_ALIGN_PARAGRAPH.RIGHT, 1.0, after=6)
        for name, key in [('DEDICATION', 'dedication'), ('ACKNOWLEDGMENTS', 'ack'), ('LIST OF TABLES', 'lot'), ('LIST OF FIGURES', 'lof'), ('LIST OF ABBREVIATIONS', 'loa')]:
            self.toc_line('', name, self.pg(('front', key)), 0, gap=4)
        p = self.doc.add_paragraph(); add_runs(p, 'CHAPTER'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, before=8, after=4)
        first_ap = next((i for i, t in enumerate(self.toc) if t[3][0] == 'ap'), len(self.toc))
        chap = self.toc[:first_ap]
        apps = self.toc[first_ap:]
        for level, num, title, key in chap:
            if level == 0:
                self.toc_line(num, title.upper(), self.pg(key), 0, gap=6)
            else:
                self.toc_line(num, title, self.pg(key), level, gap=0)
        self.toc_line('', 'REFERENCES', self.pg(('front', 'refs')), 0, gap=6)
        for level, num, title, key in apps:
            if key[0] == 'ap':
                self.toc_line('', f'{num}. {title.upper()}', self.pg(key), 0, gap=6)
            else:
                self.toc_line(num, title, self.pg(key), level, gap=0)
        self.page_break()
        self.centered('LIST OF TABLES', 12, True, after=18)
        p = self.doc.add_paragraph(); add_runs(p, 'TABLE'); p.add_run('\tPage'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, after=6)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        for num, cap in self.tabs:
            self.toc_line(num, self.resolve(cap), self.pg(('tab', num)), 1, gap=4)
        self.page_break()
        self.centered('LIST OF FIGURES', 12, True, after=18)
        p = self.doc.add_paragraph(); add_runs(p, 'FIGURE'); p.add_run('\tPage'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, after=6)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        for num, cap in self.figs:
            self.toc_line(num, self.resolve(cap), self.pg(('fig', num)), 1, gap=4)
        self.page_break()
        self.centered('LIST OF ABBREVIATIONS', 12, True, after=18)
        abbr = [('AMP', 'Automatic mixed precision'), ('BH', 'Benjamini–Hochberg (false-discovery-rate correction)'), ('CI', 'Confidence interval'),
                ('DDPM', 'Denoising diffusion probabilistic model'), ('EGNN', 'E(n)-equivariant graph neural network'), ('GIGN', 'Geometric interaction graph neural network'),
                ('HA', 'Heavy-atom count'), ('KS', 'Kolmogorov–Smirnov (test)'), ('LE', 'Ligand efficiency (Vina Dock score per heavy atom)'), ('LMDB', 'Lightning memory-mapped database'),
                ('LP split', 'Leakage-proof (target-disjoint) data split'), ('MSE', 'Mean squared error'), ('PB', 'PoseBusters physical-validity check'), ('PLIP', 'Protein–ligand interaction profiler'),
                ('pK', 'Negative decadic logarithm of an affinity constant (Ki, Kd or IC50)'), ('RA-score', 'Retrosynthetic accessibility score'), ('SA score', 'Synthetic accessibility score'),
                ('SBDD', 'Structure-based drug design'), ('Vina', 'AutoDock Vina docking program and scoring function'), ('x̂₀', 'Clean-data estimate at a diffusion step')]
        for a, b in abbr:
            p = self.doc.add_paragraph(); add_runs(p, f'{a.replace("$", "")}\t{b}'); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, left=1.5, first=-1.5, after=4)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(1.5))

    # ---- body ----
    def chapter_heading(self, label, title, kind='ch'):
        sec = self.new_section(top=1.25, bottom=1.25, fmt_='decimal', start=1 if not getattr(self, 'body_started', False) else None, first_diff=True)
        self.body_started = True
        hf_page(sec.header, WD_ALIGN_PARAGRAPH.RIGHT)
        clear_hf(sec.first_page_header); clear_hf(sec.footer)
        hf_page(sec.first_page_footer, WD_ALIGN_PARAGRAPH.CENTER)
        self.centered(('CHAPTER ' + label) if kind == 'ch' else ('APPENDIX ' + label), 12, False, before=54, after=6)
        self.centered(title.upper() if kind == 'ch' else title.upper(), 12, True, after=24)
        self.sec_seq = 0

    def heading(self, num, title, level):
        p = self.doc.add_paragraph()
        r = p.add_run(f'{num}\t'); set_font(r, 12, level == 1, level == 2)
        add_runs(p, title, 12, level == 1, level == 2)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 2.0, left=0.75 if level == 1 else 0.9, first=-0.75 if level == 1 else -0.9, before=6, after=0, keep_next=True)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75 if level == 1 else 0.9))

    def bullets(self, items):
        for it in items:
            p = self.doc.add_paragraph(); add_runs(p, '•\t' + self.resolve(it))
            fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.0, left=0.85, first=-0.25, after=12)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.85))
        # double space after list handled by after=12

    def table(self, num, key, cap, rows, note):
        p = self.doc.add_paragraph(); add_runs(p, f'Table {num}\t' + self.resolve(cap), 11)
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=18, after=8, keep_next=True, left=0.6, first=-0.6)
        p.paragraph_format.right_indent = Inches(0.0)
        ncol = len(rows[0])
        t = self.doc.add_table(rows=len(rows), cols=ncol); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        lens = [max(3, min(38, max(len(re.sub(r'[*^_{}\[\]]', '', r[c])) for r in rows))) for c in range(ncol)]
        tot = sum(lens); widths = [max(0.55, TEXT_W * l / tot) for l in lens]
        sc = TEXT_W / sum(widths); widths = [w * sc for w in widths]
        fs = 10 if ncol <= 5 else 9 if ncol <= 7 else 8
        for ri, row in enumerate(rows):
            trPr = t.rows[ri]._tr.get_or_add_trPr()
            e = OxmlElement('w:cantSplit'); trPr.append(e)
            if ri == 0:
                h = OxmlElement('w:tblHeader'); trPr.append(h)
            for ci in range(ncol):
                cell = t.cell(ri, ci); cell.width = Inches(widths[ci])
                txt = self.resolve(row[ci]) if ci < len(row) else ''
                cp = cell.paragraphs[0]; add_runs(cp, txt, fs, bold=(ri == 0))
                short = len(re.sub(r'[*^_{}]', '', txt)) < 16
                fmt(cp, WD_ALIGN_PARAGRAPH.CENTER if (ci > 0 and short) or ri == 0 else WD_ALIGN_PARAGRAPH.LEFT, 1.0, before=1, after=1)
                if ri == 0:
                    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear'); sh.set(qn('w:fill'), 'E7E9EE'); tcPr.append(sh)
        if note:
            p = self.doc.add_paragraph(); add_runs(p, self.resolve(note), 10); fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.0, before=4, after=0)
        p = self.doc.add_paragraph(); fmt(p, spacing=1.0, after=12)

    def figure(self, num, key, path, width, cap):
        full = os.path.join(os.path.dirname(HERE), path) if not os.path.isabs(path) else path
        if not os.path.exists(full):
            full = os.path.join(os.path.dirname(os.path.dirname(HERE)), path)
        if not os.path.exists(full):
            full = os.path.join(HERE, path)
        p = self.doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=12, after=6, keep_next=True)
        p.add_run().add_picture(full, width=Inches(min(width, TEXT_W)))
        c = self.doc.add_paragraph(); add_runs(c, f'Figure {num}\t' + self.resolve(cap), 11)
        fmt(c, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.0, before=6, after=24, left=0.85, first=-0.85, keep_lines=True)
        c.paragraph_format.tab_stops.add_tab_stop(Inches(0.85))

    def equation(self, num, expr, key):
        p = self.doc.add_paragraph()
        p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        add_runs(p, '\t' + self.resolve(expr) + '\t' + num, 12)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.5, before=6, after=6)

    def body(self):
        for b in self.blocks:
            k = b[0]
            if k == 'chapter':
                self.chapter_heading(b[1], b[2], 'ch')
            elif k == 'appendix':
                self.chapter_heading(b[1], b[2], 'ap')
            elif k == 'sec':
                self.heading(b[1], self.resolve(b[2]), 1)
            elif k == 'sub':
                self.heading(b[1], self.resolve(b[2]), 2)
            elif k == 'para':
                self.para(b[1])
            elif k == 'bul':
                self.bullets(b[1])
            elif k == 'tab':
                self.table(b[1], b[2], b[3], b[4], b[5])
            elif k == 'fig':
                self.figure(b[1], b[2], b[3], b[4], b[5])
            elif k == 'eq':
                self.equation(b[1], b[2], b[3])
            elif k == 'pb':
                self.page_break()

    def references(self):
        sec = self.new_section(top=1.25, bottom=1.25, fmt_='decimal', first_diff=True)
        hf_page(sec.header, WD_ALIGN_PARAGRAPH.RIGHT); clear_hf(sec.first_page_header); clear_hf(sec.footer); hf_page(sec.first_page_footer, WD_ALIGN_PARAGRAPH.CENTER)
        self.centered('REFERENCES', 12, False, before=54, after=24)
        for k in self.cite_order:
            p = self.doc.add_paragraph(); add_runs(p, f'[{self.cite_no[k]}]\t' + REFS[k]['t'], 12)
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, left=0.55, first=-0.55, after=12)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.55))

    def build(self):
        self.front()
        self.body()
        # move appendices after references: blocks list is ordered chapters..., then appendix blocks handled in body(); references inserted before first appendix
        return self


def build_docx(pages=None, path=OUT_DOCX):
    blocks, abstract = parse_sources()
    nb, xref, toc, tabs, figs = assign_numbers(blocks)
    # split appendices from chapters so REFERENCES comes between them
    idx = next((i for i, b in enumerate(nb) if b[0] == 'appendix'), len(nb))
    main, apps = nb[:idx], nb[idx:]
    bld = Builder(main + apps, abstract, xref, toc, tabs, figs, pages)
    bld.front(); bld.body()
    bld.blocks = apps
    for b in apps:                       # register appendix citations before the reference list is written
        for x in b[1:]:
            if isinstance(x, str) and '[@' in x:
                bld.resolve(x)
    bld.references()
    bld.body()
    bld.doc.core_properties.title = META['title_inline']
    bld.doc.save(path)
    unused = [k for k in REFS if k not in bld.cited]
    return bld, unused


def to_pdf(path):
    outdir = os.path.dirname(path)
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', outdir, path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return path[:-5] + '.pdf'


def pdf_pages(pdf):
    out = subprocess.run(['pdftotext', '-layout', pdf, '-'], capture_output=True, text=True).stdout
    return out.split('\f')


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def compute_pages(pdf, toc, tabs, figs):
    pages = [norm(p) for p in pdf_pages(pdf)]
    raw = pdf_pages(pdf)
    n = len([p for p in raw if p.strip()])
    # anchors
    p_ded = next(i for i, p in enumerate(raw) if re.search(r'^\s*DEDICATION\s*$', p, re.M))
    p_ch1 = next(i for i, p in enumerate(raw) if re.search(r'^\s*CHAPTER I\s*$', p, re.M))

    def disp(i):
        if i >= p_ch1:
            return str(i - p_ch1 + 1)
        v = i - p_ded + 2
        return ['', '', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii'][v]
    res = {}
    res[('front', 'dedication')] = disp(p_ded)
    for key, pat in [('ack', r'^\s*ACKNOWLEDGMENTS\s*$'), ('lot', r'^\s*LIST OF TABLES\s*$'), ('lof', r'^\s*LIST OF FIGURES\s*$'), ('loa', r'^\s*LIST OF ABBREVIATIONS\s*$')]:
        # skip the TOC page itself (entry lines carry dot leaders/page numbers) -> require heading alone and page after the TOC
        cands = [i for i, p in enumerate(raw) if i >= p_ded and re.search(pat, p, re.M) and 'TABLE OF CONTENTS' not in p]
        res[('front', key)] = disp(cands[0])
    body_start = p_ch1
    for level, num, title, key in toc:
        if key[0] == 'ch':
            pat = re.compile(r'^\s*CHAPTER ' + key[1] + r'\s*$', re.M)
            i = next(i for i in range(body_start, len(raw)) if pat.search(raw[i]))
        elif key[0] == 'ap':
            pat = re.compile(r'^\s*APPENDIX ' + key[1] + r'\s*$', re.M)
            i = next(i for i in range(body_start, len(raw)) if pat.search(raw[i]))
        else:
            needle = norm(f'{num} {re.sub(r"[*^_{}]", "", title)}')[:38]
            i = next((i for i in range(body_start, len(raw)) if needle in pages[i]), None)
            if i is None:
                needle = norm(f'{num} {title}')[:22]
                i = next((i for i in range(body_start, len(raw)) if needle in pages[i]), body_start)
        res[key] = disp(i)
    for kind, lst in (('tab', tabs), ('fig', figs)):
        for num, cap in lst:
            label = 'Table' if kind == 'tab' else 'Figure'
            needle = f'{label} {num} '
            i = next((i for i in range(body_start, len(raw)) if needle in pages[i]), body_start)
            res[(kind, num)] = disp(i)
    i_ref = next(i for i in range(body_start, len(raw)) if re.search(r'^\s*REFERENCES\s*$', raw[i], re.M))
    res[('front', 'refs')] = disp(i_ref)
    return res, n


def main():
    print('pass 1 ...', flush=True)
    bld, unused = build_docx(None)
    pdf = to_pdf(OUT_DOCX)
    blocks, _ = parse_sources(); nb, xref, toc, tabs, figs = assign_numbers(blocks)
    pages, n = compute_pages(pdf, toc, tabs, figs)
    print('pass 2 ...', flush=True)
    bld, unused = build_docx(pages)
    pdf = to_pdf(OUT_DOCX)
    pages2, n2 = compute_pages(pdf, toc, tabs, figs)
    changed = {k: (pages[k], pages2[k]) for k in pages if pages[k] != pages2[k]}
    if changed:
        print('page numbers moved after pass 2; running pass 3', len(changed), flush=True)
        bld, unused = build_docx(pages2)
        pdf = to_pdf(OUT_DOCX)
        pages3, n2 = compute_pages(pdf, toc, tabs, figs)
        print('remaining differences after pass 3:', sum(1 for k in pages2 if pages2[k] != pages3[k]))
    print(f'PDF pages: {n2}; references cited: {len(bld.cited)}; unused reference keys: {unused}')
    words = sum(len(re.findall(r"\w+", b[1])) for b in bld.blocks if b[0] == 'para')
    print('done ->', OUT_DOCX, pdf)


if __name__ == '__main__':
    main()
