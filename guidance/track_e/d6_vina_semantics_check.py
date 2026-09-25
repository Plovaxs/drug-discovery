"""Track E decision D6 (CPU only): is the `vina` column of data/affinity_info.pkl the raw (score-only) Vina score of the
stored pose, the locally-minimised score, or something else? For the 100 CrossDocked test-set reference ligands (real
receptor + SDF pose on disk) compute Vina 1.2.6 score_only and minimize and compare with the stored value."""
import json
import os
import pickle
import sys

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from utils.evaluation.docking_vina import VinaDockingTask

info = pickle.load(open('./data/affinity_info.pkl', 'rb'))
rows = []
for d in sorted(os.listdir('data/test_set')):
    for f in sorted(os.listdir(f'data/test_set/{d}')):
        if not f.endswith('.sdf'):
            continue
        key = f'{d}/{f[:-4]}'
        prot = f'data/test_set/{d}/' + f.split('_lig_')[0].rsplit('_rec', 1)[0] + '_rec.pdb'
        if key not in info or not os.path.exists(prot):
            continue
        try:
            mol = Chem.SDMolSupplier(f'data/test_set/{d}/{f}', sanitize=True)[0]
            res = {}
            for mode in ('score_only', 'minimize'):
                t = VinaDockingTask(prot, mol, tmp_dir='./tmp')
                res[mode] = float(t.run(mode=mode)[0]['affinity'])
            rows.append({'key': key, 'stored': float(info[key]['vina']), **res})
            print(len(rows), key, rows[-1], flush=True)
        except Exception as e:
            print('FAIL', key, repr(e)[:120], flush=True)

st = np.array([r['stored'] for r in rows]); so = np.array([r['score_only'] for r in rows]); mn = np.array([r['minimize'] for r in rows])
def summ(x):
    return {'mean_diff_stored_minus_calc': float((st - x).mean()), 'MAE': float(np.abs(st - x).mean()),
            'median_abs': float(np.median(np.abs(st - x))), 'frac_within_0.05': float((np.abs(st - x) < 0.05).mean()),
            'frac_within_0.5': float((np.abs(st - x) < 0.5).mean()), 'pearson': float(np.corrcoef(st, x)[0, 1])}
out = {'n': len(rows), 'vs_score_only': summ(so), 'vs_minimize': summ(mn), 'rows': rows}
json.dump(out, open('./guidance/track_e/d6_vina_semantics_results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
