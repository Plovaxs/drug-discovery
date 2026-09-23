"""Track C: scores a FINISHED, reconstructed RDKit molecule with the
frozen Stage 0 EGNN affinity model's forward pass (no gradient) -- used
purely as a post-hoc ranker for rejection sampling, never as an in-loop
guidance signal.

Deliberately uses the model's REAL training-time featurization
(utils/transforms_prop.FeaturizeLigandAtom, real Degree/NumHs/
Hybridization from RDKit-perceived bonds via
datasets.protein_ligand.get_ligand_atom_features) rather than
guidance/atom_features.py's placeholder/bond-aware approximations --
those approximations exist specifically because mid-diffusion states
have no real bonds yet; a finished, reconstructed molecule DOES have
real bonds, so there is no reason to use a lower-fidelity featurization
here. This also means Track C's ranking uses inputs from the same
distribution the model was actually trained on, unlike the guidance
gradient's input (DIAG1's traced concern).
"""
import types

import torch

from datasets.protein_ligand import get_ligand_atom_features
import utils.transforms_prop as utils_trans

_ligand_featurizer = utils_trans.FeaturizeLigandAtom()


def score_finished_molecule(model, mol, protein_pos, protein_atom_feature, device):
    """Returns a plain float: the model's raw point-estimate prediction
    (higher = predicted stronger binder, matching the pKd/pKi/pIC50
    training convention). No gradient is computed or retained."""
    ligand_element = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    ligand_atom_feature_raw = torch.tensor(get_ligand_atom_features(mol), dtype=torch.long)

    fake_data = types.SimpleNamespace(
        ligand_element=ligand_element, ligand_atom_feature=ligand_atom_feature_raw)
    fake_data = _ligand_featurizer(fake_data)
    ligand_atom_feature_full = fake_data.ligand_atom_feature_full.float().to(device)

    ligand_pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float, device=device)
    batch_ligand = torch.zeros(ligand_pos.size(0), dtype=torch.long, device=device)
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model(
            protein_pos=protein_pos, protein_atom_feature=protein_atom_feature,
            ligand_pos=ligand_pos, ligand_atom_feature=ligand_atom_feature_full,
            batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None,
        )
    return float(pred.view(-1).item())
