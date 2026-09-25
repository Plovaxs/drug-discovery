"""Track E planning-phase feasibility check (NOT training): (1) can a PLIP-compatible
pocket PDB be reconstructed from an LMDB entry (no raw PDB files on disk for training
data), and (2) do PLIP interaction counts on the reconstructed pocket match PLIP on the
real receptor? Also measures per-complex PLIP wall time."""
import json, os, sys, time, warnings, io, contextlib
warnings.filterwarnings('ignore')
import numpy as np
from datasets.pl_pair_dataset import PocketLigandPairDataset
from plip.structure.preparation import PDBComplex

from utils.data import PDBProtein
AA = {v: k for k, v in PDBProtein.AA_NAME_NUMBER.items()}
ELEM = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
OUT = sys.argv[1]
base = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')
pockets = json.load(open('guidance/task_f_pockets.json'))[:8]

def pdb_atom_line(serial, name, resname, chain, resnum, xyz, elem, het=False):
    rec = 'HETATM' if het else 'ATOM  '
    nm = name if len(name) >= 4 else (' ' + name).ljust(4)
    return f"{rec}{serial:5d} {nm} {resname:>3s} {chain}{resnum:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2s}\n"

def ligand_lines(d, start):
    out = []
    for i, (z, p) in enumerate(zip(d.ligand_element.tolist(), d.ligand_pos.numpy())):
        e = ELEM.get(z, 'C')
        out.append(pdb_atom_line(start + i, f'{e}{i+1}', 'LIG', 'L', 9999, p, e, het=True))
    return out

def reconstructed_pocket(d):
    lines, resnum, serial = [], 0, 1
    for name, aa, el, pos in zip(d.protein_atom_name, d.protein_atom_to_aa_type.tolist(), d.protein_element.tolist(), d.protein_pos.numpy()):
        if name == 'N':
            resnum += 1
        lines.append(pdb_atom_line(serial, name, AA[aa], 'A', resnum, pos, ELEM.get(el, 'C')))
        serial += 1
    return lines, serial

def run_plip(path):
    t = time.time()
    m = PDBComplex(); m.load_pdb(path)
    with contextlib.redirect_stdout(io.StringIO()):
        m.analyze()
    counts = dict(hbond=0, hydrophobic=0, pistack=0, saltbridge=0, pication=0, halogen=0, waterbridge=0, metal=0)
    for k, s in m.interaction_sets.items():
        if not k.startswith('LIG'): continue
        counts['hbond'] += len(s.hbonds_ldon) + len(s.hbonds_pdon)
        counts['hydrophobic'] += len(s.hydrophobic_contacts)
        counts['pistack'] += len(s.pistacking)
        counts['saltbridge'] += len(s.saltbridge_lneg) + len(s.saltbridge_pneg)
        counts['pication'] += len(s.pication_laro) + len(s.pication_paro)
        counts['halogen'] += len(s.halogen_bonds)
        counts['waterbridge'] += len(s.water_bridges)
        counts['metal'] += len(s.metal_complexes)
    return counts, time.time() - t

rows = []
for p in pockets:
    d = base[p['lmdb_idx']]
    real_pdb = os.path.join('data/test_set', p['expected_protein_pdb'])
    real_lines = [l for l in open(real_pdb) if l.startswith(('ATOM', 'HETATM')) ]
    pocket_lines, nxt = reconstructed_pocket(d)
    lig = ligand_lines(d, nxt)
    fr = os.path.join(OUT, f"real_{p['data_id']}.pdb"); fp = os.path.join(OUT, f"recon_{p['data_id']}.pdb")
    real_serial = len(real_lines) + 1
    open(fr, 'w').write(''.join(real_lines) + ''.join(ligand_lines(d, real_serial)) + 'END\n')
    open(fp, 'w').write(''.join(pocket_lines) + ''.join(lig) + 'END\n')
    try:
        cr, tr = run_plip(fr)
        cp, tp = run_plip(fp)
        rows.append((p['target'], cr, tr, cp, tp, len(d.ligand_element), len(d.protein_pos)))
        print(f"{p['target']:28s} real={cr} ({tr:.1f}s) | recon={cp} ({tp:.1f}s)", flush=True)
    except Exception as e:
        print(p['target'], 'PLIP FAILED:', repr(e)[:200], flush=True)
json.dump([(a, b, c, d_, e, f, g) for a, b, c, d_, e, f, g in rows], open(os.path.join(OUT, 'plip_feasibility.json'), 'w'))
if rows:
    print('mean PLIP time real: %.1fs, recon: %.1fs' % (np.mean([r[2] for r in rows]), np.mean([r[4] for r in rows])))
