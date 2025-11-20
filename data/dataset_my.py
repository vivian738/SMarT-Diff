import os
import sys

from data.data_frame_parser import postprocess_label

sys.path.insert(0, os.getcwd())
import argparse
import time
from data_frame_parser import DataFrameParser
from numpytupledataset import NumpyTupleDataset
from smile_to_graph import GGNNPreprocessor

import networkx as nx
from typing import Any, Sequence
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import Descriptors
from mol_utils import mols_to_nx, smiles_to_mols
from numpy.random import RandomState
import pandas as pd
from tqdm import tqdm
import json
import pickle

BOND_TYPES = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
ATOM_SYMBOLS= {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'Br':8, 'I':9}
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
AROMATIC = {'True':0, 'False':1}



def load_dataset(dataset_path):
    suppl = Chem.SDMolSupplier(dataset_path, removeHs=False, sanitize=True)
    dataset = pd.DataFrame()
    smiles, logP, mol_weight, tpsa, hbd, ring_count = [],[],[],[],[],[]
    for i, mol in enumerate(tqdm(suppl)):
        if mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        logP.append(Descriptors.MolLogP(mol))  # LogP
        mol_weight.append(Descriptors.MolWt(mol))  # Molecular weight
        tpsa.append(Descriptors.TPSA(mol))  # Topological polar surface area
        hbd.append(Descriptors.NumHDonors(mol))  # Number of hydrogen bond donors
        ring_count.append(Descriptors.RingCount(mol))  # Number of rings

    dataset['smiles'] = smiles
    dataset['logP'] = logP
    dataset['MW'] = mol_weight
    dataset['TPSA'] = tpsa
    dataset['HBD'] = hbd
    dataset['Ring_C'] = ring_count

    dataset.to_csv('/home/yyw/yyw/vision-lstm/ICT/controlgsde/data/chembl_prop.csv')
    # for i, mol in enumerate(tqdm(suppl)):
    #     if mol is None:
    #         continue
    #     results = []
    #     mol = Chem.AddHs(mol)

    #     for atom in mol.GetAtoms():
    #         idx = atom.GetIdx()
    #         hyb = HYBRIDIZATION_TYPE_ID[str(atom.GetHybridization())]
    #         sym = ATOM_SYMBOLS[atom.GetSymbol()]
    #         deg = atom.GetDegree()
    #         vale = atom.GetImplicitValence()
    #         aro = AROMATIC[str(atom.GetIsAromatic())]
    #         results.append((idx, [sym, hyb, deg, vale, aro]))
    #     results = sorted(results)
    #     results = [v[1] for v in results]
    #     atom_feats = np.array(results).astype(np.float32)

    #     bonds = Chem.GetAdjacencyMatrix(mol, useBO=True)
    #     bonds[np.where(bonds == 1.5)] = 4
    #     bonds = scipy.sparse.csr_matrix(bonds)

    #     y = [Descriptors.MolLogP(mol), Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol), Descriptors.RingCount(mol)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='ChEMBL', choices=['ChEMBL', 'ZINC250k', 'QM9'])
    args = parser.parse_args()

    start_time = time.time()
    data_name = args.dataset

    if data_name == 'ZINC250k':
        max_atoms = 38
        path = 'data/zinc250k.csv'
        smiles_col = 'smiles'
        label_idx = 2
    elif data_name == 'QM9':
        max_atoms = 9
        path = 'data/qm9.csv'
        smiles_col = 'SMILES1'
        label_idx = 2
    elif data_name == 'ChEMBL':
        max_atoms = 48
        label_idx = 5
        smiles_col = 'Smiles'
        path = 'data/chembl_prop.csv'
    else:
        raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

    print(f'Preprocessing {data_name} data')
    df = pd.read_csv(path)
    # Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    if data_name == 'ChEMBL':
        labels = df.keys().tolist()[label_idx:28]
    else:
        labels = df.keys().tolist()[label_idx:]
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
    result = parser.parse(df, return_smiles=True)

    dataset = result['dataset']
    smiles = result['smiles']

    NumpyTupleDataset.save(f'data/{data_name.lower()}_kekulized.npz', dataset)
    print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    with open(f'data/valid_idx_{data_name.lower()}.json') as f:
        test_idx = json.load(f)

    if data_name == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    test_smiles = [smiles[i] for i in test_idx]
    nx_graphs = mols_to_nx(smiles_to_mols(test_smiles))
    print(f'Converted the test molecules into {len(nx_graphs)} graphs')

    with open(f'data/{data_name.lower()}_test_nx.pkl', 'wb') as f:
        pickle.dump(nx_graphs, f)

    print(f'Total {time.time() - start_time:.2f} sec elapsed')