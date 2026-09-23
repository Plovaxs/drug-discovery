"""LP-PDBBind ligand similarity: Dice coefficient on Morgan fingerprints,
with the tie-breaking cascade specified in the LP-PDBBind paper for
similarity==1.0 cases (radius 2 -> 4 -> 10 -> canonical SMILES identity).
"""
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

_generators = {r: rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=1024) for r in (2, 4, 10)}


def _fp(mol, radius, n_bits=1024):
    return _generators[radius].GetFingerprint(mol)


def ligand_similarity(smiles_a, smiles_b, mol_cache=None):
    """Returns a Dice similarity in [0, 0.99] U {1.0}, where 1.0 is
    reserved for genuinely identical molecules (by canonical SMILES) and
    0.99 is the LP-PDBBind-style forced cap for non-identical molecules
    that happen to tie at 1.0 on a given fingerprint radius."""
    if mol_cache is None:
        mol_cache = {}

    def get_mol(smi):
        if smi not in mol_cache:
            mol_cache[smi] = Chem.MolFromSmiles(smi)
        return mol_cache[smi]

    mol_a, mol_b = get_mol(smiles_a), get_mol(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0

    canon_a = Chem.MolToSmiles(mol_a)
    canon_b = Chem.MolToSmiles(mol_b)
    if canon_a == canon_b:
        return 1.0

    for radius in (2, 4, 10):
        sim = DataStructs.DiceSimilarity(_fp(mol_a, radius), _fp(mol_b, radius))
        if sim < 1.0:
            return sim
        # sim == 1.0 at this radius but canonical SMILES differ -> escalate
    # tied at 1.0 through radius 10 despite differing SMILES: cap at 0.99
    return 0.99


def build_similarity_cache(smiles_list):
    """Precomputes canonical-SMILES-deduplicated Morgan fingerprints at all
    three radii for a list of SMILES, for efficient pairwise comparison
    over a large set (used by build_split.py rather than calling
    ligand_similarity() pairwise with cold RDKit mol parsing each time)."""
    fps = {}
    canon = {}
    for smi in set(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon[smi] = Chem.MolToSmiles(mol)
        fps[smi] = {r: _fp(mol, r) for r in (2, 4, 10)}
    return fps, canon


def ligand_similarity_precomputed(smi_a, smi_b, fps, canon):
    if smi_a not in fps or smi_b not in fps:
        return 0.0
    if canon[smi_a] == canon[smi_b]:
        return 1.0
    for r in (2, 4, 10):
        sim = DataStructs.DiceSimilarity(fps[smi_a][r], fps[smi_b][r])
        if sim < 1.0:
            return sim
    return 0.99
