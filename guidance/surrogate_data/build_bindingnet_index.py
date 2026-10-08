"""Builds a clean index of BindingNet v1's main archive (data/bindingnet_v1/BindingNetv1/, already
extracted from BindingNetv1.tar.gz) and filters it against this project's val+test receptor PDB IDs.

BindingNet v1 is template-based: each entry is a ChEMBL compound's pose MODELED into a template PDB
structure (not a direct crystal structure of that exact compound), with an MM-GBSA-derived affinity label
embedded in the ligand .pdb file's REMARK line, e.g.:
  REMARK   nm=CHEMBL442360-1-0  ... core_RMSD = 0.600...  Affinity = Ki=120.0nM  Similarity = 0.967...  Part_fix = No
All 159,084 .pdb files use the same format and the same unit (nM), verified by direct grep over the full
archive before writing this parser (no assumption of uM/pM/mM/scientific-notation variants -- none occur).

core_RMSD and Similarity are quality signals from the template-modeling procedure (RMSD of the predicted
pose to the template's own ligand position; Tanimoto-style similarity of the modeled compound to the
template's native ligand) -- low similarity/high RMSD entries are template-based guesses far from their
anchor and may be unreliable; this script records both but does not filter on them (left for the caller).

Filtering done here: receptor PDB ID (the `pdb_template` directory name) against this project's val+test
receptor PDB IDs (guidance/surrogate_data/val_test_exclusion_set.json, computed directly from
guidance.lp_split.lp_split_loader.build_lp_splits). This is an EXACT, high-confidence filter.

NOT done here (left as a known gap, do not treat the output as fully leakage-filtered by target identity):
ChEMBL target ID vs. this project's UniProt-derived target names are two different naming schemes with no
crossref built yet -- a compound could in principle be tested against the same protein as a val/test target
under a different ChEMBL target ID than the one on this project's test pocket, if that protein also appears
as a different template PDB elsewhere. The receptor-PDB-ID filter catches exact structure reuse; it does not
catch same-protein-different-PDB-entry cases. Build the ChEMBL target <-> UniProt crossref before training.

Usage:
  python guidance/surrogate_data/build_bindingnet_index.py
"""
import json
import math
import os
import re

import pandas as pd

ROOT = './data/bindingnet_v1/BindingNetv1'
EXCLUSION = './guidance/surrogate_data/val_test_exclusion_set.json'
OUT_FULL = './guidance/surrogate_data/bindingnet_v1_index.csv'
OUT_FILTERED = './guidance/surrogate_data/bindingnet_v1_index_filtered.csv'

REMARK_RE = re.compile(
    r'nm=(?P<name>\S+)\s+.*?core_RMSD\s*=\s*(?P<rmsd>[\d.eE+-]+)\s+'
    r'Affinity\s*=\s*(?P<atype>\w+)(?P<op>[=<>])(?P<aval>[\d.eE+-]+)nM\s+'
    r'Similarity\s*=\s*(?P<sim>[\d.eE+-]+)\s+Part_fix\s*=\s*(?P<partfix>\S+)')


def parse_remark(pdb_path):
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('REMARK') and 'Affinity' in line:
                m = REMARK_RE.search(line)
                assert m, f'REMARK format not matched in {pdb_path}: {line!r}'
                return m.groupdict()
    raise AssertionError(f'no Affinity REMARK found in {pdb_path}')


def main():
    rows = []
    for pdb_template in sorted(os.listdir(ROOT)):
        tdir = os.path.join(ROOT, pdb_template)
        if not os.path.isdir(tdir):
            continue
        for target_dir in sorted(os.listdir(tdir)):
            if not target_dir.startswith('target_'):
                continue
            chembl_target = target_dir[len('target_'):]
            cdir = os.path.join(tdir, target_dir)
            for compound_dir in sorted(os.listdir(cdir)):
                chembl_compound = compound_dir
                base = f'{pdb_template}_{chembl_target}_{chembl_compound}'
                pdb_path = os.path.join(cdir, compound_dir, base + '.pdb')
                sdf_path = os.path.join(cdir, compound_dir, base + '.sdf')
                if not (os.path.exists(pdb_path) and os.path.exists(sdf_path)):
                    continue
                r = parse_remark(pdb_path)
                aval_nM = float(r['aval'])
                pk = 9.0 - math.log10(aval_nM)
                rows.append(dict(
                    pdb_template=pdb_template, chembl_target=chembl_target, chembl_compound=chembl_compound,
                    affinity_type=r['atype'], affinity_op=r['op'], affinity_nM=aval_nM, pk=pk,
                    censored=(r['op'] != '='),  # '<value' = true affinity is tighter/more potent than reported
                    core_rmsd=float(r['rmsd']), similarity=float(r['sim']), part_fix=(r['partfix'] == 'Yes'),
                    pdb_path=pdb_path, sdf_path=sdf_path))
        if len(rows) % 20000 < 50 and rows:
            print(f'...{pdb_template}: {len(rows)} rows so far', flush=True)

    df = pd.DataFrame(rows)
    print(f'Total entries: {len(df)}  unique templates: {df.pdb_template.nunique()}  '
          f'unique targets: {df.chembl_target.nunique()}  unique compounds: {df.chembl_compound.nunique()}')
    print(df['affinity_type'].value_counts())
    df.to_csv(OUT_FULL, index=False)
    print(f'Saved full index to {OUT_FULL}')

    excl = json.load(open(EXCLUSION))
    excl_rec = set(excl['val_test_rec_pdb'])
    leaked = df['pdb_template'].isin(excl_rec)
    print(f'Receptor-PDB-ID leakage vs val+test: {leaked.sum()} / {len(df)} entries '
          f'({df.loc[leaked, "pdb_template"].nunique()} distinct template IDs) -- dropped.')
    df_clean = df.loc[~leaked].reset_index(drop=True)
    df_clean.to_csv(OUT_FILTERED, index=False)
    print(f'Saved receptor-PDB-filtered index ({len(df_clean)} entries) to {OUT_FILTERED}')
    print('NOTE: target-identity (ChEMBL vs UniProt) filtering NOT done yet -- see module docstring.')


if __name__ == '__main__':
    main()
