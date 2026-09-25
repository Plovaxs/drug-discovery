"""Track E model: the EXISTING Track A GIGN + PIGNet2 network, with (a) a learned bias and free-sign scale on the
physics-sum readout (E1 primary head) or a scalar MLP readout (ablation), and (b) an optional per-ligand-atom auxiliary
PLIP head (E2) on the same shared encoder. Track A's own files are untouched."""
import torch
import torch.nn as nn

from guidance.track_a.model import GIGNPignetAffinity


class TrackEModel(GIGNPignetAffinity):
    def __init__(self, protein_atom_feature_dim, ligand_atom_feature_dim, hidden_dim=256, head='phys', n_aux=0):
        super().__init__(protein_atom_feature_dim, ligand_atom_feature_dim, hidden_dim=hidden_dim)
        assert head in ('phys', 'mlp')
        self.head = head
        self.scale = nn.Parameter(torch.tensor(0.1))   # free sign (design 2.1)
        self.bias = nn.Parameter(torch.zeros(1))
        if head == 'mlp':
            self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.n_aux = n_aux
        if n_aux > 0:
            self.aux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_aux))
        self._last_x = None

    def _embed_and_convolve(self, *args, **kwargs):
        x, pos = super()._embed_and_convolve(*args, **kwargs)
        self._last_x = x
        return x, pos

    def shared_parameters(self):
        mods = [self.protein_atom_emb, self.ligand_atom_emb, self.hil_layers]
        return [p for m in mods for p in m.parameters()]

    def forward(self, **kw):
        n_l = kw['ligand_pos'].size(0)
        out = {}
        if self.head == 'phys':
            pred, energies = super().forward(**kw)           # populates self._last_x through the override
            out['delta_z'] = self.scale * pred + self.bias
            out['energies'] = energies
            x = self._last_x
        else:
            x, _ = self._embed_and_convolve(kw['protein_pos'], kw['protein_feat'], kw['ligand_pos'], kw['ligand_feat'],
                                            kw['edge_index_intra'], kw['edge_index_inter'])
            rows = kw['edge_index_inter'][0]
            contact = torch.unique(rows[rows >= n_l])
            pooled_c = x[contact].mean(0) if contact.numel() > 0 else torch.zeros_like(x[0])
            out['delta_z'] = self.mlp(torch.cat([x[:n_l].mean(0), pooled_c])).view(1) + self.bias
        if self.n_aux > 0:
            out['aux_logits'] = self.aux(x[:n_l])
        return out
