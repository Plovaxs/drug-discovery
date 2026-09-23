"""Guided variant of scripts/sample_diffusion.py.

Forked rather than edited in place, so the original unguided baseline stays
byte-for-byte runnable for the Task F ablation. The only behavioral
difference from sample_diffusion.py is that the per-step denoising update
goes through guidance.guided_sampling.sample_diffusion_ligand_guided instead
of model.sample_diffusion directly; everything else (dataset loading,
model construction, batching, trajectory unbatching, output format) is
identical on purpose, so results stay directly comparable and diffable
against the baseline script.

With --lambda_affinity 0 --lambda_synth 0 (the defaults, and what
configs/sampling_baseline.yml sets), this script reproduces
scripts/sample_diffusion.py's output distribution exactly — see
guidance/verify_baseline_equivalence.py, which checks this bit-for-bit on
the bundled example pocket.
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from scripts.sample_diffusion import unbatch_v_traj
from guidance.guided_sampling import sample_diffusion_ligand_guided
from guidance.load_guidance import load_guidance_models


def sample_diffusion_ligand_guided_batched(
        model, data, num_samples, batch_size=16, device='cuda:0',
        num_steps=None, pos_only=False, center_pos_mode='protein',
        sample_num_atoms='prior', affinity_model=None, synth_model=None,
        lambda_affinity=0.0, lambda_synth=0.0,
        guidance_timestep_window=None,
        capture_timesteps=None, capture_list=None,
        use_amp=False):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

        # NOTE: guidance needs autograd, so this call is NOT wrapped in
        # torch.no_grad() — sample_diffusion_ligand_guided manages its own
        # grad context internally (enabled only when guidance is active).
        r = sample_diffusion_ligand_guided(
            model,
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,

            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=num_steps,
            pos_only=pos_only,
            center_pos_mode=center_pos_mode,
            affinity_model=affinity_model,
            synth_model=synth_model,
            lambda_affinity=lambda_affinity,
            lambda_synth=lambda_synth,
            guidance_timestep_window=guidance_timestep_window,
            capture_timesteps=capture_timesteps,
            capture_list=capture_list,
            use_amp=use_amp,
        )
        with torch.no_grad():
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            all_pred_pos_traj += [p for p in all_step_pos]

            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_guided')
    parser.add_argument('--lambda_affinity', type=float, default=None,
                         help='Overrides config.sample.lambda_affinity if given')
    parser.add_argument('--lambda_synth', type=float, default=None,
                         help='Overrides config.sample.lambda_synth if given')
    args = parser.parse_args()

    logger = misc.get_logger('sampling_guided')

    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    lambda_affinity = args.lambda_affinity if args.lambda_affinity is not None \
        else config.sample.get('lambda_affinity', 0.0)
    lambda_synth = args.lambda_synth if args.lambda_synth is not None \
        else config.sample.get('lambda_synth', 0.0)
    logger.info(f'lambda_affinity={lambda_affinity}, lambda_synth={lambda_synth}')

    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    dataset, subsets = get_dataset(config=ckpt['config'].data, transform=transform)
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    affinity_model, synth_model = load_guidance_models(
        config, device=args.device, lambda_affinity=lambda_affinity, lambda_synth=lambda_synth)

    data = test_set[args.data_id]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = \
        sample_diffusion_ligand_guided_batched(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            affinity_model=affinity_model,
            synth_model=synth_model,
            lambda_affinity=lambda_affinity,
            lambda_synth=lambda_synth,
        )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'lambda_affinity': lambda_affinity,
        'lambda_synth': lambda_synth,
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
