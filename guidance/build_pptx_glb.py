"""Builds PowerPoint-native 3D models (.glb) of the docked poses directly
from atom coordinates, bypassing PyMOL's COLLADA export -- pycollada/
trimesh's COLLADA loader was found to silently drop PyMOL's per-vertex
color data (every geometry came back as flat white; verified by
inspecting the loaded trimesh.Scene's material colors before building
this script). Building the mesh directly in trimesh gives full control
over per-group material color instead.

Scene composition per target:
  - Protein backbone: a thick gray tube through consecutive CA atoms,
    giving the overall fold's shape/scale without full cartoon geometry.
  - Pocket residues (within 4.5A of the ligand): full ball-and-stick,
    sky-blue carbons / standard heteroatom colors -- same palette as
    guidance/render_examples.py's PyMOL figures.
  - Ligand: full ball-and-stick, orange carbons / standard heteroatom
    colors.
Bonds are inferred by a simple distance cutoff (no chemistry library
needed) -- adequate for a visualization, not a chemical-accuracy tool.
"""
import os

import numpy as np
import trimesh
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

OUT_DIR = './guidance/pptx_3d_models'
os.makedirs(OUT_DIR, exist_ok=True)

BOND_CUTOFF = 1.9  # Angstrom, generous single-threshold covalent bond cutoff
ATOM_RADIUS = 0.32
BOND_RADIUS = 0.16
BACKBONE_RADIUS = 0.45
POCKET_CUTOFF = 4.5

ELEMENT_COLOR = {
    'C': None,  # filled in per-group (orange for ligand, skyblue for pocket)
    'N': [50, 90, 220, 255],
    'O': [220, 50, 50, 255],
    'S': [230, 200, 40, 255],
    'F': [120, 200, 120, 255],
    'CL': [120, 200, 120, 255],
    'P': [230, 140, 40, 255],
}
DEFAULT_COLOR = [180, 180, 180, 255]
LIGAND_CARBON = [235, 104, 52, 255]
POCKET_CARBON = [110, 170, 235, 255]
BACKBONE_COLOR = [150, 150, 150, 255]


def parse_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = (atom.element or atom.get_name()[0]).strip().upper()
                    atoms.append(dict(
                        coord=atom.get_coord().astype(float),
                        element=element,
                        resname=residue.get_resname(),
                        resid=residue.get_id(),
                        chain=chain.id,
                        is_aa=is_aa(residue, standard=True),
                        name=atom.get_name(),
                    ))
        break  # first model only
    return atoms


def ca_trace(atoms):
    ca = [a for a in atoms if a['is_aa'] and a['name'] == 'CA']
    ca.sort(key=lambda a: (a['chain'], a['resid'][1]))
    return ca


def infer_bonds(atoms):
    if not atoms:
        return []
    coords = np.array([a['coord'] for a in atoms])
    bonds = []
    for i in range(len(atoms)):
        d = np.linalg.norm(coords[i + 1:] - coords[i], axis=1) if i + 1 < len(atoms) else np.array([])
        for j_offset, dist in enumerate(d):
            if dist < BOND_CUTOFF:
                bonds.append((i, i + 1 + j_offset))
    return bonds


def cylinder_between(p1, p2, radius):
    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=8)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, vec / height)
    angle = np.arccos(np.clip(np.dot(z, vec / height), -1, 1))
    if np.linalg.norm(axis) > 1e-6:
        rot = trimesh.transformations.rotation_matrix(angle, axis / np.linalg.norm(axis))
    else:
        rot = np.eye(4) if angle < 1e-3 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    translate = trimesh.transformations.translation_matrix((p1 + p2) / 2)
    cyl.apply_transform(rot)
    cyl.apply_transform(translate)
    return cyl


def sphere_at(p, radius):
    sph = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    sph.apply_translation(p)
    return sph


def color_for(element, carbon_color):
    if element == 'C':
        return carbon_color
    return ELEMENT_COLOR.get(element, DEFAULT_COLOR)


def build_ball_and_stick(atoms, carbon_color):
    if not atoms:
        return None
    meshes = []
    for a in atoms:
        s = sphere_at(a['coord'], ATOM_RADIUS)
        s.visual.vertex_colors = np.tile(color_for(a['element'], carbon_color), (len(s.vertices), 1))
        meshes.append(s)
    bonds = infer_bonds(atoms)
    for i, j in bonds:
        c = cylinder_between(atoms[i]['coord'], atoms[j]['coord'], BOND_RADIUS)
        if c is None:
            continue
        mix = np.array(color_for(atoms[i]['element'], carbon_color)) if atoms[i]['element'] != 'C' \
            else np.array(color_for(atoms[j]['element'], carbon_color))
        c.visual.vertex_colors = np.tile(mix, (len(c.vertices), 1))
        meshes.append(c)
    return trimesh.util.concatenate(meshes)


def build_backbone(ca_atoms):
    if len(ca_atoms) < 2:
        return None
    meshes = []
    for a, b in zip(ca_atoms[:-1], ca_atoms[1:]):
        # Skip chain breaks (large CA-CA gaps) so the tube doesn't jump
        # across disconnected fragments.
        if np.linalg.norm(a['coord'] - b['coord']) > 4.5:
            continue
        c = cylinder_between(a['coord'], b['coord'], BACKBONE_RADIUS)
        if c is None:
            continue
        c.visual.vertex_colors = np.tile(BACKBONE_COLOR, (len(c.vertices), 1))
        meshes.append(c)
    return trimesh.util.concatenate(meshes) if meshes else None


def build_target(name, protein, ligand):
    protein_atoms = parse_atoms(protein)
    ligand_atoms = parse_atoms(ligand)

    lig_coords = np.array([a['coord'] for a in ligand_atoms])
    pocket_atoms = []
    pocket_resids_seen = set()
    for a in protein_atoms:
        if not a['is_aa']:
            continue
        key = (a['chain'], a['resid'])
        if key in pocket_resids_seen:
            continue
        d = np.linalg.norm(lig_coords - a['coord'], axis=1).min()
        if d < POCKET_CUTOFF:
            pocket_resids_seen.add(key)
    pocket_atoms = [a for a in protein_atoms if (a['chain'], a['resid']) in pocket_resids_seen]

    scene = trimesh.Scene()
    backbone = build_backbone(ca_trace(protein_atoms))
    if backbone is not None:
        scene.add_geometry(backbone, node_name='backbone')
    pocket_mesh = build_ball_and_stick(pocket_atoms, POCKET_CARBON)
    if pocket_mesh is not None:
        scene.add_geometry(pocket_mesh, node_name='pocket_residues')
    ligand_mesh = build_ball_and_stick(ligand_atoms, LIGAND_CARBON)
    if ligand_mesh is not None:
        scene.add_geometry(ligand_mesh, node_name='ligand')

    out_path = os.path.join(OUT_DIR, f'{name}.glb')
    scene.export(out_path)
    print(f'{name}: {len(pocket_atoms)} pocket atoms, {len(ligand_atoms)} ligand atoms, '
          f'{len(ca_trace(protein_atoms))} CA -> {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)')


EXAMPLES = [
    dict(name='SQHC_ALIAD', protein='./data/test_set/SQHC_ALIAD_1_631_0/1h36_A_rec.pdb',
         ligand='./guidance/example_render/SQHC_ALIAD_pose.pdb'),
    dict(name='XANLY_BACGL', protein='./data/test_set/XANLY_BACGL_26_777_0/2e24_A_rec.pdb',
         ligand='./guidance/example_render/XANLY_BACGL_pose.pdb'),
    dict(name='CD38_HUMAN', protein='./data/test_set/CD38_HUMAN_44_300_0/3dzh_A_rec.pdb',
         ligand='./guidance/example_render/CD38_HUMAN_pose.pdb'),
    dict(name='HDAC8_HUMAN', protein='./data/test_set/HDAC8_HUMAN_1_377_0/4rn0_B_rec.pdb',
         ligand='./guidance/example_render/HDAC8_pose.pdb'),
    dict(name='P2Y12_HUMAN', protein='./data/test_set/P2Y12_HUMAN_1_342_0/4pxz_A_rec.pdb',
         ligand='./guidance/example_render/P2Y12_pose.pdb'),
]

if __name__ == '__main__':
    for ex in EXAMPLES:
        build_target(**ex)
