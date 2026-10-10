"""Regression tests for the BindingNet pocket-extraction primitives.

Why these exist specifically: both behaviours tested here have ALREADY broken this project once.

  1. AMBER residue variants. The pilot was written after inspecting a single template (`10gs`), whose
     receptor happened to use only standard residue names, and the code was generalised from that sample
     of one. Other templates use HID/HIE/CYM, and extraction died with `KeyError: 'HID'` partway through
     a long run. The fix (AA_VARIANTS) is cheap; what is worth protecting is the GUARANTEE, because a
     regression here does not fail fast -- it fails hours in, on template N of 5,758.

  2. Hydrogen stripping. Pocket selection is driven by residue centers of mass. If hydrogens survive the
     strip, every center of mass shifts and the selected pocket residues change, silently producing
     training data subtly inconsistent with the CrossDocked pockets the model is validated on. That is
     the dangerous class of bug: no exception, just quietly different data.

Pure-text unit tests with hand-written PDB fixtures -- no receptor files, no network, no GPU -- so they
run on every commit even while training occupies the GPU.

Usage:
  PYTHONPATH=. python guidance/surrogate_data/tests/test_pocket_extraction.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from guidance.surrogate_data.extract_bindingnet_pockets import (AA_VARIANTS, element_of,
                                                                normalize_and_strip)
from utils.data import PDBProtein


def atom_line(serial=1, name=' CA ', res_name='ALA', chain='A', res_seq=1,
              x=0.0, y=0.0, z=0.0, element='C'):
    """A column-exact PDB ATOM record. Built by absolute column offsets on purpose: the production code
    slices by column (line[17:20], line[76:78]), so a fixture assembled by casual concatenation would
    test nothing."""
    # Column layout per the PDB spec, 0-indexed: serial 6:11, name 12:16, altLoc 16:17,
    # resName 17:20, chainID 21:22, resSeq 22:26, x/y/z 30:38/38:46/46:54, element 76:78.
    # The altLoc blank at 16 is easy to omit and shifts resName by one if you do -- which is exactly
    # the mistake this fixture made on first writing, so test_fixture_is_column_correct guards it.
    return (f'ATOM  {serial:>5} {name:<4} {res_name:>3} {chain}{res_seq:>4}    '
            f'{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}')


def atoms_of(pdb_text):
    return [l for l in pdb_text.splitlines() if l.startswith('ATOM')]


# -------------------------------------------------------- the fixture itself

def test_fixture_is_column_correct():
    """The fixture is the measuring instrument for every other test here; if its columns are off, the
    column assertions below pass or fail for reasons unrelated to the production code. Checked against
    the PDB spec rather than against the production parser, so the two cannot agree on a shared error."""
    line = atom_line(serial=42, name=' CA ', res_name='HIS', chain='B', res_seq=137,
                     x=1.5, y=-2.25, z=3.125, element='C')
    assert line[0:6] == 'ATOM  '
    assert int(line[6:11]) == 42
    assert line[12:16] == ' CA '
    assert line[16] == ' ', 'altLoc column must be blank, not the first char of resName'
    assert line[17:20] == 'HIS'
    assert line[21] == 'B'
    assert int(line[22:26]) == 137
    assert float(line[30:38]) == 1.5
    assert float(line[38:46]) == -2.25
    assert float(line[46:54]) == 3.125
    assert line[76:78] == ' C'
    assert len(line) == 78


# ---------------------------------------------------------------- element_of

def test_element_read_from_columns_77_78():
    assert element_of(atom_line(element='C')) == 'C'
    assert element_of(atom_line(element='N')) == 'N'
    assert element_of(atom_line(element='O')) == 'O'
    assert element_of(atom_line(element='S')) == 'S'


def test_two_letter_element_is_capitalized_not_uppercased():
    # 'FE' must become 'Fe', matching PDBProtein's .capitalize(). Uppercasing instead yields a symbol
    # no periodic-table lookup recognises.
    assert element_of(atom_line(element='FE')) == 'Fe'


def test_element_falls_back_to_atom_name_when_field_blank():
    line = atom_line(name=' CB ', element='')
    assert line[76:78].strip() == ''
    assert element_of(line) == 'C'


def test_hydrogen_detected_from_either_source():
    assert element_of(atom_line(name=' HA ', element='H')) == 'H'
    assert element_of(atom_line(name=' HA ', element='')) == 'H'


def test_element_of_agrees_with_pdbprotein_inference():
    """The contract is not 'returns something sensible' but 'agrees with PDBProtein', since the two must
    partition the same atoms identically."""
    for name, elem in [(' CA ', 'C'), (' N  ', 'N'), (' O  ', 'O'), (' SD ', 'S'),
                       (' CB ', ''), (' HA ', ''), (' FE ', 'FE')]:
        line = atom_line(name=name, element=elem)
        expected = line[76:78].strip().capitalize() or line[13:14]
        assert element_of(line) == expected, name


# ------------------------------------------------------- normalize_and_strip

def test_hydrogens_are_removed():
    text = '\n'.join([atom_line(1, ' CA ', 'ALA', element='C'),
                      atom_line(2, ' HA ', 'ALA', element='H'),
                      atom_line(3, ' CB ', 'ALA', element='C')]) + '\n'
    out, unknown = normalize_and_strip(text)
    kept = atoms_of(out)
    assert len(kept) == 2
    assert all(element_of(l) != 'H' for l in kept)
    assert unknown == set()


def test_hydrogens_without_element_column_are_also_removed():
    """BindingNet's `rec_h_opt.pdb` files are AMBER output; trusting the element column alone would leave
    hydrogens in for any file that omits it."""
    out, _ = normalize_and_strip(atom_line(1, ' HB1', 'ALA', element='') + '\n')
    assert atoms_of(out) == []


def test_every_amber_variant_maps_to_its_parent():
    for variant, parent in sorted(AA_VARIANTS.items()):
        out, unknown = normalize_and_strip(atom_line(1, ' CA ', variant, element='C') + '\n')
        line = atoms_of(out)[0]
        assert line[17:20].strip() == parent, f'{variant} -> {line[17:20]!r}, expected {parent}'
        assert unknown == set(), f'{variant} should be mapped, not reported unknown'


def test_every_mapped_parent_is_known_to_pdbprotein():
    """The point of the mapping is that the OUTPUT is featurizable. A mapping onto a name PDBProtein
    still does not know would merely relocate the KeyError."""
    for variant, parent in sorted(AA_VARIANTS.items()):
        assert parent in PDBProtein.AA_NAME_NUMBER, f'{variant}->{parent} absent from AA_NAME_NUMBER'


def test_the_variants_that_broke_production_are_still_covered():
    # Named explicitly so a future edit that trims AA_VARIANTS cannot silently reintroduce the original
    # `KeyError: 'HID'` crash.
    for v in ('HID', 'HIE', 'HIP', 'CYM', 'CYX'):
        assert v in AA_VARIANTS, f'{v} removed from AA_VARIANTS -- reintroduces a known crash'


def test_residue_rewrite_preserves_column_alignment():
    """Residue names are patched by slicing. A parent written without .ljust(3) padding would shift every
    downstream column and corrupt the coordinates -- which would not raise, it would just be wrong."""
    original = atom_line(1, ' CA ', 'HID', x=1.5, y=-2.25, z=3.125, element='C')
    out, _ = normalize_and_strip(original + '\n')
    line = atoms_of(out)[0]
    assert len(line) == len(original)
    for lo, hi in [(30, 38), (38, 46), (46, 54), (76, 78), (6, 11), (12, 16), (21, 22), (22, 26)]:
        assert line[lo:hi] == original[lo:hi], f'columns {lo}:{hi} shifted'


def test_unknown_residue_is_skipped_and_reported_not_raised():
    text = '\n'.join([atom_line(1, ' CA ', 'ALA', element='C'),
                      atom_line(2, ' CA ', 'XYZ', res_seq=2, element='C')]) + '\n'
    out, unknown = normalize_and_strip(text)
    kept = atoms_of(out)
    assert len(kept) == 1
    assert kept[0][17:20].strip() == 'ALA'
    assert unknown == {'XYZ'}


def test_non_atom_records_pass_through_untouched():
    text = ('HEADER    TEST\nSEQRES   1 A    1  ALA\n'
            + atom_line(1, ' CA ', 'ALA', element='C') + '\nEND\n')
    out, _ = normalize_and_strip(text)
    for line in ('HEADER    TEST', 'SEQRES   1 A    1  ALA', 'END'):
        assert line in out


def test_hetatm_is_not_treated_as_atom():
    """normalize_and_strip inspects only records whose name is exactly 'ATOM'; HETATM lines pass through.
    Asserted here so the behaviour is a recorded decision rather than an accident."""
    het = 'HETATM' + atom_line(1, ' O  ', 'HOH', element='O')[6:]
    out, unknown = normalize_and_strip(het + '\n')
    assert het in out
    assert unknown == set()


def test_output_always_ends_with_newline():
    out, _ = normalize_and_strip(atom_line(1, ' CA ', 'ALA', element='C'))
    assert out.endswith('\n')


def test_empty_input_does_not_raise():
    out, unknown = normalize_and_strip('')
    assert unknown == set()
    assert out.strip() == ''


# ----------------------------------------- the consequence hydrogens have

def _residue_center(text):
    protein = PDBProtein(normalize_and_strip(text)[0])
    assert len(protein.residues) == 1
    return np.asarray(protein.residues[0]['center_of_mass'], dtype=np.float64)


def test_hydrogens_do_not_shift_residue_center_of_mass():
    """The reason hydrogen stripping matters, asserted end-to-end rather than by inspection: the residue
    center of mass is what query_residues_ligand thresholds on, so it must not move when hydrogens are
    present in the input."""
    heavy = [atom_line(1, ' N  ', 'ALA', x=0.0, element='N'),
             atom_line(2, ' CA ', 'ALA', x=1.0, element='C'),
             atom_line(3, ' C  ', 'ALA', x=2.0, element='C')]
    with_h = heavy + [atom_line(4, ' HA ', 'ALA', x=50.0, element='H'),
                      atom_line(5, ' HB1', 'ALA', x=-50.0, element='H')]
    a = _residue_center('\n'.join(heavy) + '\n')
    b = _residue_center('\n'.join(with_h) + '\n')
    # Hydrogens placed 50 A out: a regressed strip fails this by a wide margin, not by FP noise.
    assert np.abs(a - b).max() < 1e-9, f'center moved by {np.abs(a - b).max()}'


def test_amber_variant_does_not_shift_center_of_mass_either():
    """HID vs HIS must give an identical heavy-atom representation -- that is the stated justification for
    calling the mapping lossless."""
    def block(res):
        return '\n'.join([atom_line(1, ' N  ', res, x=0.0, element='N'),
                          atom_line(2, ' CA ', res, x=1.0, element='C'),
                          atom_line(3, ' CB ', res, x=2.5, element='C')]) + '\n'
    assert np.abs(_residue_center(block('HID')) - _residue_center(block('HIS'))).max() < 1e-12


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
