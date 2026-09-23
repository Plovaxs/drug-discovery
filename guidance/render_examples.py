"""Thesis/paper illustration figures: re-dock the best PoseBusters-valid
molecule per selected pocket to recover its real docked pose, then render
a publication-quality PyMOL figure. One-off qualitative script, not part
of the statistical pipeline.

v3 (publication pass) changes vs. v2 (guidance/render_pymol.py's fixes):
  - Ligand/pocket-residue carbon colors switched to this project's
    validated dataviz palette (guidance/plot_track_c_figures.py's
    CAT_ORANGE / CAT_BLUE) for visual consistency with the Track C
    statistical figures in the same thesis.
  - Key H-bond partner residues are auto-detected (polar atom within 3.5A
    of a ligand polar atom) and labeled with resn+resi -- readers no
    longer have to guess which residue a dashed line points to.
  - Resolution raised to 2400x1800 (was 1800x1350) at ray_trace_mode=0
    (kept from v2 -- mode 1 caused the black-patch artifacts).
  - Combined 1x3 composite panel (matplotlib) for a single "representative
    binding poses" figure, labeled A/B/C to match the Track C statistical
    figure's panel-labeling convention.
"""
import os
import subprocess

import pymol2
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.evaluation.docking_vina import VinaDockingTask

PROTEIN_ROOT = './data/test_set'
OUT_DIR = './guidance/example_render'
os.makedirs(OUT_DIR, exist_ok=True)

def _hex_to_rgb01(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# Matches guidance/plot_track_c_figures.py's validated palette.
LIGAND_CARBON = _hex_to_rgb01('#eb6834')   # CAT_ORANGE
POCKET_CARBON = _hex_to_rgb01('#2a78d6')   # CAT_BLUE
HBOND_COLOR = _hex_to_rgb01('#eda100')     # yellow -- kept close to the
                                           # near-universal structural-
                                           # biology convention for H-bonds

EXAMPLES = [
    dict(name='SQHC_ALIAD', protein_pdb='SQHC_ALIAD_1_631_0/1h36_A_rec.pdb',
         ligand_sdf='./guidance/track_c_pools/pocket70/seed2021/sdf/259.sdf',
         expected_score=-15.786),
    dict(name='XANLY_BACGL', protein_pdb='XANLY_BACGL_26_777_0/2e24_A_rec.pdb',
         ligand_sdf='./guidance/track_c_pools/pocket45/seed2021/sdf/235.sdf',
         expected_score=-14.089),
    dict(name='CD38_HUMAN', protein_pdb='CD38_HUMAN_44_300_0/3dzh_A_rec.pdb',
         ligand_sdf='./guidance/track_c_pools/pocket10/seed2021/sdf/173.sdf',
         expected_score=-13.002),
]


def redock(example):
    protein_path = os.path.join(PROTEIN_ROOT, example['protein_pdb'])
    mol = next(iter(Chem.SDMolSupplier(example['ligand_sdf'], sanitize=True)))
    assert mol is not None, f"failed to load {example['ligand_sdf']}"

    print(f"Re-docking {example['name']} ...")
    task = VinaDockingTask(protein_path, mol, tmp_dir='./tmp')
    result = task.run(mode='dock', exhaustiveness=8)
    score = result[0]['affinity']
    print(f"  score={score:.3f} (expected ~{example['expected_score']})")

    pdbqt_path = os.path.join(OUT_DIR, f"{example['name']}_pose.pdbqt")
    with open(pdbqt_path, 'w') as f:
        f.write(result[0]['pose'])
    pdb_path = os.path.join(OUT_DIR, f"{example['name']}_pose.pdb")
    subprocess.run(['obabel', pdbqt_path, '-O', pdb_path], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return protein_path, pdb_path, score


def render(protein_path, ligand_pdb, out_png, score=None, label_title=None):
    p = pymol2.PyMOL()
    p.start()
    cmd = p.cmd

    cmd.set_color('ligand_c', LIGAND_CARBON)
    cmd.set_color('pocket_c', POCKET_CARBON)
    cmd.set_color('hbond_c', HBOND_COLOR)

    cmd.load(protein_path, 'prot')
    cmd.load(ligand_pdb, 'lig')

    cmd.bg_color('white')
    cmd.hide('everything')

    cmd.show('cartoon', 'prot')
    cmd.color('gray85', 'prot')
    cmd.set('cartoon_transparency', 0.55, 'prot')
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_highlight_color', 'grey70')

    cmd.show('sticks', 'lig')
    cmd.set('stick_radius', 0.22, 'lig')
    cmd.color('ligand_c', 'lig and elem C')
    cmd.color('atomic', 'lig and not elem C')

    cmd.select('pocket_res', 'byres (prot within 4.5 of lig)')
    cmd.show('sticks', 'pocket_res and sidechain')
    cmd.set('stick_radius', 0.12, 'pocket_res')
    cmd.color('pocket_c', 'pocket_res and elem C')
    cmd.color('atomic', 'pocket_res and not elem C')

    cmd.distance('hbonds', 'lig', 'pocket_res', mode=2)
    cmd.hide('labels', 'hbonds')
    cmd.color('hbond_c', 'hbonds')
    cmd.set('dash_width', 3.5)
    cmd.set('dash_gap', 0.3)
    cmd.set('dash_radius', 0.06)

    # Auto-label the specific residues actually forming a polar contact
    # with the ligand, rather than leaving the reader to guess which
    # sidechain a dashed line belongs to.
    cmd.select('hbond_partners',
               'byres (pocket_res and (name N*+O*) within 3.5 of (lig and name N*+O*))')
    cmd.label('hbond_partners and name CA', '"%s%s" % (resn, resi)')
    cmd.set('label_size', 20)
    cmd.set('label_color', 'black')
    cmd.set('label_outline_color', 'white')
    cmd.set('label_font_id', 7)  # sans-serif
    cmd.set('label_bg_color', 'white')

    cmd.set('ray_trace_mode', 0)
    cmd.set('ray_shadows', 0)
    cmd.set('two_sided_lighting', 1)
    cmd.set('ambient', 0.4)
    cmd.set('direct', 0.6)
    cmd.set('reflect', 0.3)
    cmd.set('antialias', 2)
    cmd.set('depth_cue', 0)

    cmd.orient('lig or pocket_res')
    cmd.zoom('lig or pocket_res', buffer=3)
    cmd.turn('x', -10)

    cmd.ray(2400, 1800)
    cmd.png(out_png, dpi=300)
    p.stop()


def make_composite(rendered):
    """rendered: list of (name, png_path, score). Builds a labeled 1x3
    panel composite matching the Track C statistical figure's A/B/C
    labeling convention (guidance/plot_track_c_figures.py)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, len(rendered), figsize=(6 * len(rendered), 6))
    panel_labels = ['A', 'B', 'C', 'D', 'E']
    for ax, (name, png_path, score), label in zip(axes, rendered, panel_labels):
        img = mpimg.imread(png_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{label}. {name}   (Vina Dock = {score:.2f})',
                     fontsize=13, fontweight='bold', loc='left', color='#0b0b0b')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'composite_binding_poses')
    fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote composite -> {out_path}.png / .pdf')


def main():
    rendered = []
    for ex in EXAMPLES:
        protein_path, ligand_pdb, score = redock(ex)
        out_png = os.path.join(OUT_DIR, f"{ex['name']}_render.png")
        render(protein_path, ligand_pdb, out_png, score=score)
        print(f"  rendered -> {out_png}")
        rendered.append((ex['name'], out_png, score))
    make_composite(rendered)


if __name__ == '__main__':
    main()
