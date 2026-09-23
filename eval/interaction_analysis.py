"""Protein-ligand interaction analysis for Task G blueprint reports.

Builds on structures already produced by the pipeline -- no new structure
prediction happens here, only geometric analysis of a pose that already
exists.

Note on which pose is analyzed: this module analyzes the diffusion-
generated 3D ligand pose (the conformer already in the RDKit Mol coming
out of reconstruction), not a Vina-re-optimized pose. The version of
meeko pinned by this environment (0.1.dev3, per the repo's own README) has
no supported PDBQT-to-RDKit round trip that reliably preserves atom
identity/order (Vina's 'dock' mode pose comes back as a raw PDBQT string
with explicit hydrogens added by meeko's ligand preparation, which does
not map 1:1 back onto the original heavy-atom RDKit Mol without a fragile
re-matching step) -- so rather than silently approximating that mapping,
this deliberately analyzes the actual pose the diffusion model produced.
The Vina Dock score reported alongside it (from honest_eval.py) is still
computed by Vina in the usual way; only the *geometry* used for contact
analysis here is the generated pose, not Vina's internally re-docked one.

Primary path: PLIP (Protein-Ligand Interaction Profiler), which gives
*typed* interactions -- H-bonds, hydrophobic contacts, pi-stacking, salt
bridges, halogen bonds, water bridges -- each with a distance and the
protein residue involved. PLIP needs the receptor and ligand in one PDB
file (ligand as a HETATM group); write_complex_pdb builds that from the
receptor PDB (already on disk) and the docked ligand pose (an RDKit Mol
with 3D coordinates).

Fallback: if PLIP fails to parse the complex or detect the ligand (this can
happen depending on how the receptor PDB is formatted), or PLIP is not
importable, this falls back to a simple distance-based contact map: any
receptor residue with an atom within `cutoff` Angstrom of any ligand atom,
parsed via Biopython (receptor) and RDKit (ligand pose). This fallback
carries NO interaction typing (no H-bond/hydrophobic/etc. classification)
-- every returned contact is labeled with which method produced it
(`source`: 'plip' or 'distance_fallback') so the two confidence levels are
never blurred together in a report.

All output is phrased as geometric fact about the docked pose, never
physiological effect -- see guidance in eval/blueprint_report.py.
"""
import os
import tempfile

from rdkit import Chem


PLIP_INTERACTION_LABELS = {
    'hbond': 'hydrogen bond',
    'hydrophobic': 'hydrophobic contact',
    'pistacking': 'pi-stacking',
    'pication': 'pi-cation interaction',
    'saltbridge': 'salt bridge',
    'halogenbond': 'halogen bond',
    'waterbridge': 'water bridge (mediated)',
}


def write_complex_pdb(protein_path, ligand_mol, out_path, ligand_chain='X'):
    """Merges a receptor PDB and a docked ligand pose into one PDB file
    PLIP can parse (ligand as a distinctly-chained HETATM group).
    """
    mol = Chem.AddHs(Chem.Mol(ligand_mol), addCoords=True)
    ligand_block = Chem.MolToPDBBlock(mol)
    ligand_lines = []
    for line in ligand_block.splitlines():
        if line.startswith('HETATM') or line.startswith('ATOM'):
            # Force HETATM + a distinct chain ID (column 22) so PLIP's
            # ligand finder treats this group as the ligand, not protein.
            line = 'HETATM' + line[6:]
            line = line[:21] + ligand_chain + line[22:]
            ligand_lines.append(line)

    with open(protein_path) as f:
        protein_lines = [l.rstrip('\n') for l in f
                         if l.startswith('ATOM') or l.startswith('HETATM') or l.startswith('TER')]

    with open(out_path, 'w') as f:
        f.write('\n'.join(protein_lines) + '\n')
        f.write('TER\n')
        f.write('\n'.join(ligand_lines) + '\n')
        f.write('TER\nEND\n')


def _plip_interactions(complex_pdb_path):
    """Returns a list of contact dicts via PLIP, or raises on any failure
    (caller is responsible for catching and falling back).
    """
    from plip.structure.preparation import PDBComplex

    complex_ = PDBComplex()
    complex_.load_pdb(complex_pdb_path)
    if not complex_.ligands:
        raise ValueError('PLIP found no ligand in the complex PDB')
    complex_.analyze()

    contacts = []
    for site_id, interaction_set in complex_.interaction_sets.items():
        groups = {
            'hbond': list(interaction_set.hbonds_ldon) + list(interaction_set.hbonds_pdon),
            'hydrophobic': list(interaction_set.hydrophobic_contacts),
            'pistacking': list(interaction_set.pistacking),
            'pication': list(interaction_set.pication_laro) + list(interaction_set.pication_paro),
            'saltbridge': list(interaction_set.saltbridge_lneg) + list(interaction_set.saltbridge_pneg),
            'halogenbond': list(interaction_set.halogen_bonds),
            'waterbridge': list(interaction_set.water_bridges),
        }
        for itype, items in groups.items():
            for it in items:
                distance = getattr(it, 'distance', None)
                if distance is None:
                    distance = getattr(it, 'distance_ad', None)
                contacts.append({
                    'source': 'plip',
                    'interaction_type': PLIP_INTERACTION_LABELS[itype],
                    'chain': getattr(it, 'reschain', None),
                    'resname': getattr(it, 'restype', None),
                    'resnum': getattr(it, 'resnr', None),
                    'distance_A': round(float(distance), 2) if distance is not None else None,
                })
    return contacts


def _distance_fallback(protein_path, ligand_mol, cutoff=4.0):
    """Distance-only contact map: receptor residues with any atom within
    `cutoff` Angstrom of any ligand atom. No interaction typing.
    """
    from Bio.PDB import PDBParser, NeighborSearch

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('receptor', protein_path)
    receptor_atoms = [a for a in structure.get_atoms()]
    ns = NeighborSearch(receptor_atoms)

    conf = ligand_mol.GetConformer()
    ligand_coords = [conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())]

    closest_by_residue = {}
    for pos in ligand_coords:
        nearby = ns.search([pos.x, pos.y, pos.z], cutoff, level='A')
        for atom in nearby:
            residue = atom.get_parent()
            if residue.id[0] != ' ':  # skip het/water residues in the receptor itself
                continue
            key = (residue.get_parent().id, residue.get_resname(), residue.id[1])
            d = ((atom.coord[0] - pos.x) ** 2 + (atom.coord[1] - pos.y) ** 2 + (atom.coord[2] - pos.z) ** 2) ** 0.5
            if key not in closest_by_residue or d < closest_by_residue[key]:
                closest_by_residue[key] = d

    contacts = []
    for (chain, resname, resnum), dist in sorted(closest_by_residue.items(), key=lambda kv: kv[1]):
        contacts.append({
            'source': 'distance_fallback',
            'interaction_type': None,  # no typing available from this method
            'chain': chain,
            'resname': resname,
            'resnum': resnum,
            'distance_A': round(dist, 2),
        })
    return contacts


def analyze_interactions(protein_path, ligand_mol, cutoff=4.0, verbose=False):
    """Returns (contacts, method) where method is 'plip' or 'distance_fallback'."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            complex_path = os.path.join(tmpdir, 'complex.pdb')
            write_complex_pdb(protein_path, ligand_mol, complex_path)
            contacts = _plip_interactions(complex_path)
        if contacts:
            return contacts, 'plip'
        raise ValueError('PLIP returned zero contacts')
    except Exception as e:
        if verbose:
            print(f'[interaction_analysis] PLIP failed ({e}); '
                  f'falling back to distance-only contact map (cutoff={cutoff} A).')
        return _distance_fallback(protein_path, ligand_mol, cutoff=cutoff), 'distance_fallback'
