"""Exports the same docked-pose scenes used in guidance/render_examples.py
as PowerPoint-native 3D models (.glb), so they can be dropped into a slide
via PowerPoint's Insert > 3D Models and rotated/zoomed live during a
thesis defense, fully offline (PowerPoint 2016+/365 supports .glb, .fbx,
.obj, .3mf, .ply, .stl natively).

Pipeline: PyMOL renders the scene (protein cartoon, ligand sticks, pocket
residue sticks -- same styling as the static figures) and exports it as
COLLADA (.dae, PyMOL open-source's own supported 3D export format), which
trimesh then converts to glTF binary (.glb; PyMOL cannot write .glb
directly, and gltf export needs an external collada2gltf binary that
isn't installed here -- trimesh sidesteps that).
"""
import os

import pymol2
import trimesh

OUT_DIR = './guidance/pptx_3d_models'
os.makedirs(OUT_DIR, exist_ok=True)

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


def build_scene(cmd, protein_path, ligand_path):
    cmd.load(protein_path, 'prot')
    cmd.load(ligand_path, 'lig')
    cmd.hide('everything')

    cmd.show('cartoon', 'prot')
    cmd.color('gray80', 'prot')
    cmd.set('cartoon_transparency', 0.0)  # PowerPoint's viewer handles opacity poorly; keep opaque

    cmd.show('sticks', 'lig')
    cmd.set('stick_radius', 0.25, 'lig')
    cmd.color('orange', 'lig and elem C')
    cmd.color('atomic', 'lig and not elem C')

    cmd.select('pocket_res', 'byres (prot within 4.5 of lig)')
    cmd.show('sticks', 'pocket_res and sidechain')
    cmd.set('stick_radius', 0.15, 'pocket_res')
    cmd.color('skyblue', 'pocket_res and elem C')
    cmd.color('atomic', 'pocket_res and not elem C')

    cmd.orient('lig or pocket_res')
    cmd.zoom('lig or pocket_res', buffer=4)


def export_one(example):
    p = pymol2.PyMOL()
    p.start()
    cmd = p.cmd
    build_scene(cmd, example['protein'], example['ligand'])

    dae_path = os.path.join(OUT_DIR, f"{example['name']}.dae")
    cmd.save(dae_path)
    p.stop()

    scene = trimesh.load(dae_path)
    glb_path = os.path.join(OUT_DIR, f"{example['name']}.glb")
    scene.export(glb_path)
    os.remove(dae_path)

    size_kb = os.path.getsize(glb_path) / 1024
    print(f"{example['name']}: wrote {glb_path} ({size_kb:.0f} KB)")


def main():
    for ex in EXAMPLES:
        export_one(ex)


if __name__ == '__main__':
    main()
