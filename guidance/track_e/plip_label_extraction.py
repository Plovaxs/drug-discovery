"""Track E planning-phase utilities (feasibility only, no training).

Rebuild a PLIP-compatible complex from an LMDB entry (raw PDB/SDF files are not on disk for the
CrossDocked training data) and extract ligand-atom-level interaction labels with PLIP.

Label classes (see TRACK_E_DESIGN.md 2.1): hbond, hydrophobic, pi_stack (pi-stacking + pi-cation),
salt_bridge. Halogen bonds, water bridges and metal complexes are dropped (too sparse / not
reconstructible from the pocket-only LMDB).
"""
import contextlib
import io
import warnings

import numpy as np

warnings.filterwarnings('ignore')
from plip.structure.preparation import PDBComplex
from utils.data import PDBProtein

AA = {v: k for k, v in PDBProtein.AA_NAME_NUMBER.items()}
ELEM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
CLASSES = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']


def _line(serial, name, resname, chain, resnum, xyz, elem, het):
    rec = 'HETATM' if het else 'ATOM  '
    nm = name if len(name) >= 4 else (' ' + name).ljust(4)
    return (f"{rec}{serial:5d} {nm} {resname:>3s} {chain}{resnum:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2s}\n")


def build_complex_pdb(d, ligand_pos=None):
    """Returns (pdb_text, first ligand serial). Residues start at each backbone 'N'."""
    lines, resnum, serial = [], 0, 1
    for name, aa, el, pos in zip(d.protein_atom_name, d.protein_atom_to_aa_type.tolist(),
                                 d.protein_element.tolist(), d.protein_pos.numpy()):
        if name == 'N':
            resnum += 1
        lines.append(_line(serial, name, AA[aa], 'A', resnum, pos, ELEM.get(el, 'C'), False))
        serial += 1
    lig_start = serial
    lpos = d.ligand_pos.numpy() if ligand_pos is None else ligand_pos
    for i, (z, p) in enumerate(zip(d.ligand_element.tolist(), lpos)):
        e = ELEM.get(z, 'C')
        lines.append(_line(lig_start + i, f'{e}{i + 1}', 'LIG', 'L', 9999, p, e, True))
    return ''.join(lines) + 'END\n', lig_start


def ligand_atom_labels(pdb_text, lig_start, ligand_elements, tmp_path):
    """Returns ((n_lig, 4) uint8 label matrix, n_element_mismatches).

    hbond/hydrophobic atoms are addressed by PDB serial; ring/charge groups by 1-based index in the
    ligand molecule. The element check verifies that the latter mapping lands on the right atom.
    """
    n_lig = len(ligand_elements)
    with open(tmp_path, 'w') as f:
        f.write(pdb_text)
    m = PDBComplex()
    m.load_pdb(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        m.analyze()
    lab = np.zeros((n_lig, len(CLASSES)), dtype=np.uint8)
    mismatches = 0

    def mark_serial(serial, cls):
        j = int(serial) - lig_start
        if 0 <= j < n_lig:
            lab[j, CLASSES.index(cls)] = 1

    def mark_group(atoms, cls):
        nonlocal mismatches
        for a in atoms:
            j = a.idx - 1
            if 0 <= j < n_lig:
                if a.atomicnum != ligand_elements[j]:
                    mismatches += 1
                lab[j, CLASSES.index(cls)] = 1

    for key, s in m.interaction_sets.items():
        if not key.startswith('LIG'):
            continue
        for h in s.hbonds_ldon:
            mark_serial(h.d_orig_idx, 'hbond')
        for h in s.hbonds_pdon:
            mark_serial(h.a_orig_idx, 'hbond')
        for h in s.hydrophobic_contacts:
            mark_serial(h.ligatom_orig_idx, 'hydrophobic')
        for p in s.pistacking:
            mark_group(p.ligandring.atoms, 'pi_stack')
        for p in s.pication_laro:
            mark_group(p.ring.atoms, 'pi_stack')
        for p in s.pication_paro:
            mark_group(p.charge.atoms, 'pi_stack')
        for sb in s.saltbridge_lneg:
            mark_group(sb.negative.atoms, 'salt_bridge')
        for sb in s.saltbridge_pneg:
            mark_group(sb.positive.atoms, 'salt_bridge')
    return lab, mismatches
