"""Part 1 diagnostic: why did lambda=1.0 dual guidance fragment the output?

Two checks, both on the bundled example pocket:
  1. Reconstruct the lambda=1.0 dual-guidance sample and inspect fragments
     via RDKit GetMolFrags -- how many pieces, how big is the largest.
  2. At one representative sampling step, compare grad_pos.norm() from
     each guidance model against the diffusion core's own base-score
     magnitude at that step, to see whether guidance gradients are simply
     too large relative to the base signal.
"""
import sys

import torch

import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from scripts.sample_for_pocket import pdb_to_pocket_data
from guidance.guided_sampling import sample_diffusion_ligand_guided
from guidance.affinity_guidance import AffinityGuidance
from guidance.synth_guidance import SynthGuidance
from utils import reconstruct
from rdkit import Chem

EXAMPLE_PDB = 'examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb'
CHECKPOINT = 'pretrained_models/pretrained_diffusion.pt'


def load_everything(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer])
    data = pdb_to_pocket_data(EXAMPLE_PDB)
    data = transform(data).to(device)
    model = ScorePosNet3D(
        ckpt['config'].model, protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    affinity_model = AffinityGuidance('./guidance_models/affinity_egnn.pt', device=device)
    synth_model = SynthGuidance('./guidance_models/synth_ra_score.pt', device=device)
    return data, model, affinity_model, synth_model


def check_1_fragments(device='cuda:0', n_atoms=22, seed=777, lambda_affinity=1.0, lambda_synth=1.0):
    print(f'=== Check 1: fragment inspection at lambda_affinity={lambda_affinity}, '
          f'lambda_synth={lambda_synth} ===')
    data, model, affinity_model, synth_model = load_everything(device)
    protein_v = data.protein_atom_feature.float()

    misc.seed_all(seed)
    batch_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=device)
    batch_ligand = torch.zeros(n_atoms, dtype=torch.long, device=device)
    center = data.protein_pos.mean(dim=0, keepdim=True)
    init_pos = center + torch.randn(n_atoms, 3, device=device)
    init_v = log_sample_categorical(torch.zeros(n_atoms, model.num_classes, device=device))

    r = sample_diffusion_ligand_guided(
        model, protein_pos=data.protein_pos, protein_v=protein_v, batch_protein=batch_protein,
        init_ligand_pos=init_pos.clone(), init_ligand_v=init_v.clone(), batch_ligand=batch_ligand,
        num_steps=1000, center_pos_mode='protein', pos_only=False,
        affinity_model=affinity_model, synth_model=synth_model,
        lambda_affinity=lambda_affinity, lambda_synth=lambda_synth,
    )

    pred_atom_type = trans.get_atomic_number_from_index(r['v'].cpu(), mode='add_aromatic')
    pred_aromatic = trans.is_aromatic_from_index(r['v'].cpu(), mode='add_aromatic')
    pos_list = r['pos'].cpu().numpy().astype(float).tolist()
    try:
        mol = reconstruct.reconstruct_from_generated(
            pos_list, [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
    except Exception as e:
        print('reconstruction raised (already a bad sign):', e)
        return

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    print(f'n_fragments: {len(frags)}')
    sizes = sorted([f.GetNumAtoms() for f in frags], reverse=True)
    print(f'fragment sizes (heavy+H atoms): {sizes}')
    for i, f in enumerate(frags[:3]):
        try:
            smi = Chem.MolToSmiles(f, canonical=False)
        except Exception:
            smi = '(could not get SMILES, likely unsanitizable)'
        print(f'  fragment {i} ({f.GetNumAtoms()} atoms): {smi}')
    print()
    return len(frags), sizes


def check_2_gradient_magnitudes(device='cuda:0', n_atoms=22, seed=777, check_step_frac=0.5):
    print('=== Check 2: gradient magnitude vs base-score magnitude ===')
    data, model, affinity_model, synth_model = load_everything(device)
    protein_v = data.protein_atom_feature.float()

    misc.seed_all(seed)
    batch_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=device)
    batch_ligand = torch.zeros(n_atoms, dtype=torch.long, device=device)
    center = data.protein_pos.mean(dim=0, keepdim=True)
    ligand_pos = center + torch.randn(n_atoms, 3, device=device)
    ligand_v = log_sample_categorical(torch.zeros(n_atoms, model.num_classes, device=device))

    num_steps = model.num_timesteps
    check_step = int(num_steps * check_step_frac)
    time_seq = list(reversed(range(0, num_steps)))

    with torch.no_grad():
        for i in time_seq:
            t = torch.full((1,), fill_value=i, dtype=torch.long, device=device)
            preds = model(protein_pos=data.protein_pos, protein_v=protein_v, batch_protein=batch_protein,
                         init_ligand_pos=ligand_pos, init_ligand_v=ligand_v, batch_ligand=batch_ligand,
                         time_step=t)
            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']

            if i == check_step:
                base_pos_norm = pos0_from_e.norm().item()
                base_pos_mean_abs = pos0_from_e.abs().mean().item()
                with torch.enable_grad():
                    pos0_req = pos0_from_e.detach().requires_grad_(True)
                    v0_req = v0_from_e.detach().requires_grad_(True)
                    g_aff_pos, g_aff_v = affinity_model.grad_log_score(
                        pos0_req, v0_req, batch_ligand, data.protein_pos, protein_v, batch_protein)
                    g_syn_pos, g_syn_v = synth_model.grad_log_score(
                        pos0_req, v0_req, batch_ligand, data.protein_pos, protein_v, batch_protein)
                print(f'step {i} (t frac={check_step_frac}):')
                print(f'  base pos0 norm: {base_pos_norm:.4f}  (mean abs: {base_pos_mean_abs:.4f})')
                print(f'  affinity grad_pos norm: {g_aff_pos.norm().item():.4f}  '
                     f'(mean abs: {g_aff_pos.abs().mean().item():.4f})')
                print(f'  synth    grad_pos norm: {g_syn_pos.norm().item():.4f}  '
                     f'(mean abs: {g_syn_pos.abs().mean().item():.4f})')
                print(f'  ratio affinity_grad/base: {g_aff_pos.norm().item() / max(base_pos_norm, 1e-8):.4f}')
                print(f'  ratio synth_grad/base: {g_syn_pos.norm().item() / max(base_pos_norm, 1e-8):.4f}')
                break

            # advance one step unguided (this is just a probe -- we only need to reach check_step)
            from models.molopt_score_model import index_to_log_onehot
            pos_model_mean = model.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = model.posterior_logvar[t][batch_ligand].unsqueeze(-1)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)

            import torch.nn.functional as F
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, model.num_classes)
            log_model_prob = model.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v = log_sample_categorical(log_model_prob)
    print()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    check_1_fragments(device=device)
    check_2_gradient_magnitudes(device=device)
