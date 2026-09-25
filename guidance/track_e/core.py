"""Track E core pieces whose silent failure would invalidate the whole track (unit-tested in tests/):
  * the frozen delta-target definition and its reconstruction
  * the scrambled-PLIP-label negative control permutation
  * the anchor table (idx -> target/pk/vina/n_lig) and the cached PLIP label store
"""
import glob
import os
import pickle

import numpy as np

KCAL_PER_PK = 1.364  # RT ln10 at 298 K
CACHE_DIR = './guidance/track_e/cache'
PLIP_CLASSES_ALL = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']
PRIMARY_CLASSES = ['hbond', 'hydrophobic', 'salt_bridge']  # decision D2: pi_stack excluded from the primary head


# ---------------------------------------------------------------- delta target (pre-registered, Sec. 2.3) ----
def pk_vina(vina):
    """Anchor: pK_Vina = max(-vina, 0) / 1.364. Positive (clashing) Vina scores are clipped to 0."""
    return np.maximum(-np.asarray(vina, dtype=np.float64), 0.0) / KCAL_PER_PK


def delta_target(pk, vina):
    """Delta = pK_exp - pK_Vina."""
    return np.asarray(pk, dtype=np.float64) - pk_vina(vina)


def reconstruct(vina, delta_hat):
    """yhat = pK_Vina + Delta_hat (both in pK units)."""
    return pk_vina(vina) + np.asarray(delta_hat, dtype=np.float64)


# ---------------------------------------------------------------- scrambled-label negative control ----------
def scramble_permutation(n_lig, seed):
    """Returns src[i]: index of the complex whose label matrix complex i receives. Complexes are permuted only
    among those with the SAME ligand atom count (so label matrices keep their shape and per-size statistics);
    within each group of size >= 2 a random cyclic shift guarantees no complex keeps its own labels. Groups of
    size 1 keep their own labels (returned so the caller can report how many were left unscrambled)."""
    n_lig = np.asarray(n_lig)
    rng = np.random.RandomState(seed)
    src = np.arange(len(n_lig))
    for n in np.unique(n_lig):
        members = np.where(n_lig == n)[0]
        if len(members) < 2:
            continue
        order = members[rng.permutation(len(members))]
        src[order] = np.roll(order, 1)
    return src


# ---------------------------------------------------------------- anchor table --------------------------------
def build_anchor_table(force=False):
    """(idx, target, pk, vina, n_lig) for every complex of the LP split (train/val/test in file order)."""
    path = os.path.join(CACHE_DIR, 'anchor_table.npz')
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from guidance.lp_split.lp_split_loader import build_lp_splits
    info = pickle.load(open('./data/affinity_info.pkl', 'rb'))
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    out = {}
    for part in ('train', 'val', 'test'):
        rows = []
        for i in splits[part]:
            d = base[i]
            rows.append((i, d.protein_filename.split('/')[0], pk_by_idx[i],
                         float(info[d.ligand_filename[:-4]]['vina']), len(d.ligand_element)))
        out[part] = np.array(rows, dtype=object)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **out)
    return out


def split_arrays(table, part):
    r = table[part]
    return dict(idx=r[:, 0].astype(int), target=r[:, 1].astype(str), pk=r[:, 2].astype(float),
                vina=r[:, 3].astype(float), n_lig=r[:, 4].astype(int))


# ---------------------------------------------------------------- PLIP label store ----------------------------
class PlipLabelStore:
    """idx -> (n_lig, 4) uint8 ligand-atom label matrix (classes PLIP_CLASSES_ALL), read from cache chunks."""

    def __init__(self, cache_dir=os.path.join(CACHE_DIR, 'plip')):
        self.labels = {}
        for f in sorted(glob.glob(os.path.join(cache_dir, 'chunk_*.npz'))):
            z = np.load(f)
            idx, lens, flat = z['idx'], z['lens'], z['flat']
            off = 0
            for i, n in zip(idx, lens):
                self.labels[int(i)] = flat[off:off + n * 4].reshape(n, 4)
                off += n * 4

    def __contains__(self, i):
        return int(i) in self.labels

    def get(self, i, classes=PRIMARY_CLASSES):
        cols = [PLIP_CLASSES_ALL.index(c) for c in classes]
        return self.labels[int(i)][:, cols]
