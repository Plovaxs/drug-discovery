"""Headless PyMOL render of the example docked pose (guidance/render_example_pose.py
must be run first). One-off thesis illustration figure, not part of the
statistical pipeline."""
import pymol2

PROTEIN_PDB = './data/test_set/SQHC_ALIAD_1_631_0/1h36_A_rec.pdb'
LIGAND_PDB = './guidance/example_render/best_pose.pdb'
OUT_PNG = './guidance/example_render/best_pose_render.png'

p = pymol2.PyMOL()
p.start()
cmd = p.cmd

cmd.load(PROTEIN_PDB, 'protein')
cmd.load(LIGAND_PDB, 'ligand')

cmd.bg_color('white')
cmd.hide('everything')

cmd.show('cartoon', 'protein')
cmd.color('gray80', 'protein')
cmd.set('cartoon_transparency', 0.3, 'protein')

cmd.show('sticks', 'ligand')
cmd.color('yellow', 'ligand and elem C')
cmd.color('atomic', 'ligand and not elem C')
cmd.set('stick_radius', 0.2, 'ligand')

cmd.select('pocket_residues', 'byres (protein within 4.5 of ligand)')
cmd.show('sticks', 'pocket_residues and not name C+N+O')
cmd.show('lines', 'pocket_residues')
cmd.color('cyan', 'pocket_residues and elem C')
cmd.color('atomic', 'pocket_residues and not elem C')

cmd.distance('hbonds', 'ligand', 'pocket_residues', mode=2)
cmd.hide('labels', 'hbonds')
cmd.color('red', 'hbonds')

cmd.orient('ligand')
cmd.zoom('ligand', buffer=6)

cmd.set('ray_opaque_background', 0)
cmd.set('antialias', 2)
cmd.set('ray_trace_mode', 1)
cmd.set('ray_shadows', 0)

cmd.ray(1600, 1200)
cmd.png(OUT_PNG, dpi=300)

print(f'Rendered to {OUT_PNG}')
p.stop()
