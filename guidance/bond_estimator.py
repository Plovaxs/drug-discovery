"""Phase 1 of the architectural-upgrade addendum: differentiable, bond-aware
Degree/NumHs/Hybridization estimation from the diffusion state, replacing
the fixed "unknown" placeholders in guidance/atom_features.py.

Root cause being addressed: the affinity/synth guidance models were
*trained* on real molecules where Degree/NumHs/Hybridization come from
actual RDKit-perceived bonds (datasets/protein_ligand.py's
get_ligand_atom_features). At *inference* (guidance/atom_features.py, prior
to this module), those three fields were fixed to a constant "unknown"
one-hot placeholder, because the diffusion process doesn't have discrete
bonds at all -- only atom positions and types. That train/inference
mismatch was diagnosed (guidance/SYNTH_GUIDANCE_FINDING.md) as the likely
cause of the synth guidance model's own score and the real, independently
computed RA-score moving in *opposite* directions as guidance strength
increased: the guidance model had no reason to behave sensibly on inputs
unlike anything in its training distribution.

Approach (1a from the addendum -- parameter-free, no extra training run):
estimate a *soft* (continuous, differentiable) bond order between every
atom pair from interatomic distance and covalent-radius-based reference
bond lengths -- reusing the exact same bonds1/bonds2/bonds3 covalent-length
tables already in utils/evaluation/analyze.py (used there for the hard,
non-differentiable stability/validity check) -- then derive soft
Degree/NumHs from that soft adjacency, and a soft "unsaturation" proxy for
Hybridization. Continuous quantities are turned into the fixed-width
one-hot-shaped slots FeaturizeLigandAtom expects via GaussianSmearing
(models/common.py) -- the same smooth-binning building block already used
elsewhere in this codebase for continuous-to-vector featurization, so nothing
new needs training and everything stays differentiable end to end.

This is a real approximation, not a claim of exact chemistry: sigmoid-soft
bond detection can misjudge ambiguous cases (e.g. a genuinely stretched
single bond vs. a compressed non-bond), and the Hybridization proxy in
particular only weakly resembles RDKit's true classification. It is
offered as strictly better than a constant "unknown" placeholder for every
atom regardless of local geometry, not as ground truth. See Phase 1's
sweep checkpoint in SYNTH_GUIDANCE_FINDING.md for whether it actually
helps.
"""
import torch
import torch.nn as nn

from models.common import GaussianSmearing
from utils.evaluation.analyze import bonds1, bonds2, bonds3, atom_decoder, margin1, margin2, margin3

# Element order must match guidance/atom_features.py's ELEMENT_ORDER /
# utils/transforms_prop.FeaturizeLigandAtom.atomic_numbers.
_ELEMENT_ORDER = [1, 6, 7, 8, 9, 15, 16, 17]
_STEEPNESS = 0.2  # sigmoid steepness (1/pm) for the soft within-threshold indicator
# NOTE: this was originally 4.0 and, checked empirically, saturated the
# sigmoid within a fraction of a picometer of the threshold -- i.e. behaved
# as a hard step with ~zero gradient almost everywhere, defeating the point
# of a differentiable estimator. 0.2/pm gives a ~20pm-wide soft transition
# band (sigmoid moves from 0.1 to 0.9 over about +-11pm around the
# threshold), comparable to real bond-length variability, and was verified
# to produce a nonzero pos.grad on a realistic (non-idealized) geometry.


def _bond_length_tensor(order_table, device):
    """(8, 8) tensor of reference bond lengths (pm) for `order_table`
    (bonds1/2/3), indexed by position in _ELEMENT_ORDER; -1 (no such bond
    order exists for that pair, e.g. H-H triple) becomes +inf so the soft
    indicator for it is always ~0.
    """
    n = len(_ELEMENT_ORDER)
    table = torch.full((n, n), float('inf'))
    for i, zi in enumerate(_ELEMENT_ORDER):
        for j, zj in enumerate(_ELEMENT_ORDER):
            si, sj = atom_decoder[zi], atom_decoder[zj]
            length = order_table.get(si, {}).get(sj, -1)
            if length is not None and length > 0:
                table[i, j] = float(length)
    return table.to(device)


class BondAwareFeatureEstimator(nn.Module):
    """Parameter-free (no training required) differentiable estimator of
    Degree/NumHs/Hybridization-proxy features from geometry, to replace
    guidance/atom_features.py's fixed placeholders.
    """

    def __init__(self, hybridization_dim, degree_classes=6, numhs_classes=6, device='cuda:0'):
        super().__init__()
        self.degree_classes = degree_classes
        self.numhs_classes = numhs_classes
        self.hybridization_dim = hybridization_dim
        self.device = device

        self.len1 = _bond_length_tensor(bonds1, device)
        self.len2 = _bond_length_tensor(bonds2, device)
        self.len3 = _bond_length_tensor(bonds3, device)

        # Smooth binning: continuous soft-degree/soft-numhs (roughly 0-5)
        # -> a `classes`-wide vector, same trick FeaturizeProteinAtom-style
        # modules elsewhere use to turn a scalar into a fixed-width slot.
        self.degree_smearing = GaussianSmearing(start=0, stop=degree_classes - 1,
                                                num_gaussians=degree_classes, fixed_offset=False).to(device)
        self.numhs_smearing = GaussianSmearing(start=0, stop=numhs_classes - 1,
                                               num_gaussians=numhs_classes, fixed_offset=False).to(device)
        # Unsaturation proxy: 0 (fully saturated, sp3-like) to ~2 (sp-like,
        # two pi bonds) -- binned into whatever width the real
        # Hybridization one-hot occupies in the target feature layout.
        self.hybridization_smearing = GaussianSmearing(start=0, stop=2.0,
                                                        num_gaussians=hybridization_dim, fixed_offset=False).to(device)

    def soft_bond_order(self, pos, element_idx):
        """pos: (N, 3) Angstrom. element_idx: (N,) long, index into
        _ELEMENT_ORDER (NOT atomic number). Returns (N, N) soft bond order
        in [0, 3], diagonal zeroed.
        """
        n = pos.size(0)
        dist_pm = torch.cdist(pos, pos) * 100.0  # Angstrom -> pm
        len1 = self.len1[element_idx][:, element_idx]
        len2 = self.len2[element_idx][:, element_idx]
        len3 = self.len3[element_idx][:, element_idx]

        within1 = torch.sigmoid(_STEEPNESS * (len1 + margin1 - dist_pm))
        # len2/len3 are +inf for element pairs with no such bond order at all
        # (e.g. C-H has no double/triple bond) -- sigmoid(inf * finite) = 1
        # regardless of actual distance, so those must be masked to exactly
        # 0 rather than left to the sigmoid, or every such pair would
        # silently inherit within1's value as if a double/triple bond were
        # always present.
        has2 = torch.isfinite(len2)
        has3 = torch.isfinite(len3)
        within2 = torch.sigmoid(_STEEPNESS * (len2 + margin2 - dist_pm)) * within1 * has2
        within3 = torch.sigmoid(_STEEPNESS * (len3 + margin3 - dist_pm)) * within2 * has3

        order = within1 + within2 + within3
        order = order * (1 - torch.eye(n, device=pos.device))  # no self-bonds
        return order

    def forward(self, pos, atomic_numbers):
        """pos: (N, 3). atomic_numbers: (N,) long, real atomic numbers
        (e.g. 6 for carbon). Returns (degree_feat, numhs_feat,
        hybridization_feat), each (N, respective_classes), ready to
        concatenate into the same slot FeaturizeLigandAtom's placeholder
        one-hots occupied.
        """
        z_to_idx = {z: i for i, z in enumerate(_ELEMENT_ORDER)}
        element_idx = torch.tensor([z_to_idx.get(int(z), 0) for z in atomic_numbers], device=pos.device)

        order = self.soft_bond_order(pos, element_idx)  # (N, N)
        soft_degree = order.sum(dim=1).clamp(max=self.degree_classes - 1)  # count of ~single-bond-equivalents

        is_h = (atomic_numbers == 1).float()
        soft_numhs = (order * is_h.view(1, -1)).sum(dim=1).clamp(max=self.numhs_classes - 1)

        # unsaturation proxy: bond order beyond a plain single bond, summed
        # per atom (0 for a saturated sp3 center, higher for sp2/sp)
        pair_excess = (order - torch.sigmoid(_STEEPNESS * (self.len1[element_idx][:, element_idx] + margin1
                                                            - torch.cdist(pos, pos) * 100.0))).clamp(min=0)
        unsaturation = pair_excess.sum(dim=1).clamp(max=2.0)

        degree_feat = self.degree_smearing(soft_degree)
        numhs_feat = self.numhs_smearing(soft_numhs)
        hybridization_feat = self.hybridization_smearing(unsaturation)
        return degree_feat, numhs_feat, hybridization_feat
