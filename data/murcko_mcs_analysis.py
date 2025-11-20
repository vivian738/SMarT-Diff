import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw

def load_smiles(filename):
    df = pd.read_csv(filename)
    return [Chem.MolFromSmiles(sm) for sm in df.iloc[:, 0].dropna() if Chem.MolFromSmiles(sm)]

def get_bm_scaffolds(mols):
    scaffolds = []
    for mol in mols:
        try:
            scf = MurckoScaffold.GetScaffoldForMol(mol)
            if scf is not None:
                scaffolds.append(scf)
        except:
            continue
    return scaffolds

def unique_scaffold_set(scaffolds):
    smiles_set = set()
    unique = []
    for mol in scaffolds:
        smi = Chem.MolToSmiles(mol)
        if smi not in smiles_set:
            unique.append(mol)
            smiles_set.add(smi)
    return unique

def compute_mcs(mol_a, mol_b):
    mcs_result = rdFMCS.FindMCS([mol_a, mol_b])
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    return mcs_mol, mcs_result

# 主流程
if __name__ == "__main__":
    # 文件路径
    gsk3b_file = "gsk3b_jnk3_pdb/gsk3b.csv"
    jnk3_file = "gsk3b_jnk3_pdb/jnk3.csv"

    # 加载分子
    gsk3b_mols = load_smiles(gsk3b_file)
    jnk3_mols  = load_smiles(jnk3_file)

    # 提取骨架
    gsk3b_scfs = unique_scaffold_set(get_bm_scaffolds(gsk3b_mols))
    jnk3_scfs  = unique_scaffold_set(get_bm_scaffolds(jnk3_mols))

    mcs_smiles_set = set()
    for i in gsk3b_scfs:
        for j in jnk3_scfs:
            mcs_mol, mcs_result = compute_mcs(i, j)
            if mcs_mol and mcs_result.numAtoms > 6:
                try:
                    mcs_smiles_set.add(mcs_result.smartsString)
                except:
                    continue
    pd.DataFrame(list(mcs_smiles_set)).to_csv("gsk3b_jnk3_pdb/gsk3b_jnk3.csv", index=False, header=False)
    print('Dual targets: Extracted MCS of BM scaffold are finished')

