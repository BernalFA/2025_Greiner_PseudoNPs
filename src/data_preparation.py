import sys
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Lipinski import HeavyAtomCount
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

sys.path.append(str(Path.cwd()))

from src.utils import standardize_mol, CompoundLibraryFilter, count_nitrogen_atoms  # noqa: E402


def prepare_chembl_with_sugars(chembl: pd.DataFrame) -> pd.DataFrame:
    df = chembl.copy()
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles")

    pandarallel.initialize(nb_workers=24, progress_bar=True)
    df["clean_mol"] = df["ROMol"].parallel_apply(standardize_mol)
    df["clean_smiles"] = df["clean_mol"].apply(Chem.MolToSmiles)
    return df

# Interestingly, CDK mols are left as some sort of radicals (apparently, the mol2smi transformation
# does not work well and the implicit hydrogens are somehow treated as radical sources).
# The smiles have to be manually modified
def remove_false_radicals(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    new_smi = Chem.MolToSmiles(mol)  # necessary to avoid conflict with mutable objects in pandas
    return new_smi


# At this point, sugars were removed using the CDK Sugar Remover extension for KNIME
def prepare_chembl_no_sugars() -> pd.DataFrame:
    chembl_no_sugars = pd.read_csv(Path.cwd() / "data" / "interim" / "chembl_35_NP_no_sugars.csv")
    chembl_no_sugars.fillna("", inplace=True)
    for _, row in chembl_no_sugars.iterrows():
        if row["smiles_no_sugar"] == "":
            row["smiles_no_sugar"] = row["clean_smiles"]

    PandasTools.AddMoleculeColumnToFrame(chembl_no_sugars, smilesCol="smiles_no_sugar")

    pandarallel.initialize(nb_workers=24, progress_bar=True)
    chembl_no_sugars["no_radical_smi"] = chembl_no_sugars["ROMol"].parallel_apply(remove_false_radicals)
    PandasTools.AddMoleculeColumnToFrame(chembl_no_sugars, smilesCol="no_radical_smi",
                                        molCol="no_radical_mol")
    chembl_no_sugars["clean_mol"] = chembl_no_sugars["no_radical_mol"].parallel_apply(standardize_mol)

    compound_filter = CompoundLibraryFilter(mol_col="clean_mol")
    chembl_no_sugars_filtered = compound_filter.filter(chembl_no_sugars)
    chembl_no_sugars_filtered["taut_mol"] = chembl_no_sugars_filtered["clean_mol"].parallel_apply(
        standardize_mol, canonicalize_tautomer=True
    )
    chembl_no_sugars_filtered["taut_smiles"] = chembl_no_sugars_filtered["taut_mol"].apply(Chem.MolToSmiles)
    return chembl_no_sugars_filtered


def process_library(df: pd.DataFrame, smilesCol: str="smiles", molCol: str="ROMol",
                    workers: int=12, subsample: bool=False) -> pd.DataFrame:
    df = df.copy()
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smilesCol, molCol=molCol)
    df.dropna(subset=[molCol], inplace=True)
    if subsample:
        df = df.sample(50_000, random_state=2025)

    pandarallel.initialize(nb_workers=workers, progress_bar=True)
    df["taut_mol"] = df[molCol].parallel_apply(
        standardize_mol, canonicalize_tautomer=True
    )
    compound_filter = CompoundLibraryFilter(mol_col="taut_mol")
    df_filtered = compound_filter.filter(df)
    df_filtered["taut_smiles"] = df_filtered["taut_mol"].apply(Chem.MolToSmiles)
    result = df_filtered[["ID", "taut_smiles"]].copy()
    return result


def get_scaffolds(representatives: dict) -> set:
    scaffolds_smiles = set()
    for smi in representatives.values():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)
        if scaffold_smi not in scaffolds_smiles:
            scaffolds_smiles.add(scaffold_smi)
    return scaffolds_smiles


def contain_min_scaffold(mol: Chem.Mol, family: str) -> bool:
    if family == "mias":
        mias = {
            "tryptamine": "C1=CC=C2C(=C1)C(=CN2)CCN",
            "indoline-ethanamine": "C1C(C2=CC=CC=C2N1)CCN",
            "indoline-propanamine": "C1=CC=C2C(=C1)C(=CN2)CCCN",
            "indolenine": "C2=Nc1ccccc1C2"
        }
        scaffolds = mias.values()
    elif family == "amaryllidaceae":
        representatives_am = {
            "Lycorine_ed": "C1CN2CC3=CC=C(C=C3[C@H]4[C@H]2C1=C[C@@H]([C@H]4O)O)",
            "Galanthamine": "CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CC(=C34)C1)OC)O",
            "Tazettine_ed": "CN1C[C@@]2([C@]3([C@@H]1C[C@@H](C=C3)OC)C4=CC=C(C=C4CO2))O",
            "Narciclasine_ed": "C2=CC(=C3C(=C2)C4=C[C@@H]([C@H]([C@H]([C@@H]4NC3=O)O)O)O)O",
            "Montanine_ed": "CO[C@H]1C=C2[C@H](C[C@@H]1O)N3C[C@H]2C4=CC=C(C=C4C3)",
            "Lycorenine": "CN1CCC2=CC[C@@H]3[C@H]([C@@H]21)C4=CC(=C(C=C4[C@H](O3)O)OC)OC",
            "Haemanthamine_ed": "CO[C@H]1C[C@H]2[C@@]3(C=C1)[C@H](CN2CC4=CC=C(C=C34))O",
            "Crinine_ed": "C1CN2CC3=CC=C(C=C3[C@]15[C@H]2C[C@H](C=C5)O)",
            "Gracilmine_ed": "[H][C@]12C3CC[C@@H]4NCC[C@@]41C1=C(C=CC=C1)[C@@]2([H])NC3"
        }
        scaffolds = get_scaffolds(representatives_am)
    elif family == "hasubanan":
        hasubanan = {
            "hasubanan": "C2C[C@]14CCCC[C@@]4(CCN1)c3ccccc23",
        }
        scaffolds = hasubanan.values()
    if any(mol.HasSubstructMatch(Chem.MolFromSmiles(query)) for query in scaffolds):
        return True
    return False


def comply_restrictions_mias(mol: Chem.Mol) -> bool:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    n_heavy_atoms = HeavyAtomCount(scaffold)
    sssr = Chem.GetSSSR(mol)
    if n_heavy_atoms < 16 or n_heavy_atoms > 28:
        return False
    if count_nitrogen_atoms(scaffold) != 2:
        return False
    if len(sssr) <= 3:
        return False
    return True


def comply_restrictions_amaryllidaceae(mol: Chem.Mol) -> bool:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if count_nitrogen_atoms(scaffold) > 1:
        return False
    return True


def prepare_mias(database_df: pd.DataFrame) -> pd.DataFrame:
    mias = []
    for i, row in tqdm(database_df.iterrows(), total=len(database_df)):
        mol = Chem.MolFromSmiles(row["taut_smiles"])
        if contain_min_scaffold(mol, family="mias"):
            if comply_restrictions_mias(mol):
                mias.append(i)

    return database_df.iloc[mias].copy()


def prepare_sceletium() -> pd.DataFrame:
    full_sceletium = {
        "ID": [
            "Mesembrine",
            "Mesembrenone",
            "D7-Mesembrenone",
            "Mesembranol",
            "4'-O-demethylmesembranol",
            "Mesembrenol",
            "4'-O-demethylmesembrenol",
            "4'-O-demethylmesembrenone",
            "Sceletenone",
            "N-demethyl-N-formyl-mesembrenone",
            "O-acetylmesembrenol",
            "Mesembrane",
            "N-demethylmesembranol",
            "N-demethylmesembrenol",
            "Sceletium A4",
        ],
        "smiles": [
            "CN1CC[C@]2([C@@H]1CC(=O)CC2)C3=CC(=C(C=C3)OC)OC",
            "CN1CC[C@]2([C@@H]1CC(=O)C=C2)C3=CC(=C(C=C3)OC)OC",
            "CN1CCC2(C1=CC(=O)CC2)C3=CC(=C(C=C3)OC)OC",
            "CN1CCC2(C1CC(CC2)O)C3=CC(=C(C=C3)OC)OC",
            "CN1CCC2(C1CC(CC2)O)C3=CC(=C(C=C3)O)OC",
            "CN1CCC2(C1CC(C=C2)O)C3=CC(=C(C=C3)OC)OC",
            "CN1CCC2(C1CC(C=C2)O)C3=CC(=C(C=C3)O)OC",
            "CN1CC[C@]2([C@@H]1CC(=O)C=C2)C3=CC(=C(C=C3)O)OC",
            "CN1CC[C@]2([C@@H]1CC(=O)C=C2)C3=CC=C(C=C3)O",
            "O=CN1CC[C@]2([C@@H]1CC(=O)C=C2)C3=CC(=C(C=C3)OC)OC",
            "CN1CCC2(C1CC(C=C2)OC(=O)C)C3=CC(=C(C=C3)OC)OC",
            "CN1CC[C@@]2([C@H]1CCCC2)C3=CC(=C(C=C3)OC)OC",
            "N1CCC2(C1CC(CC2)O)C3=CC(=C(C=C3)OC)OC",
            "N1CCC2(C1CC(C=C2)O)C3=CC(=C(C=C3)OC)OC",
            "COC1=CC=C([C@@]23CCC4=NC=CC=C4[C@@H]2N(C)CC3)C=C1OC",
        ]
    }

    sceletium = pd.DataFrame(full_sceletium)
    return sceletium


def prepare_amaryllidaceae(database_df: pd.DataFrame) -> pd.DataFrame:
    am_alkaloids = []
    for i, row in tqdm(database_df.iterrows(), total=len(database_df)):
        mol = Chem.MolFromSmiles(row["taut_smiles"])
        if contain_min_scaffold(mol, family="amaryllidaceae"):
            if comply_restrictions_amaryllidaceae(mol):
                am_alkaloids.append(i)

    return database_df.iloc[am_alkaloids].copy()


def prepare_hasubanan(database_df: pd.DataFrame) -> pd.DataFrame:
    hasubanan = []
    for i, row in tqdm(database_df.iterrows(), total=len(database_df)):
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            if contain_min_scaffold(mol, family="hasubanan"):
                hasubanan.append(i)
    
    hasubanan = database_df.iloc[hasubanan].copy()
    PandasTools.AddMoleculeColumnToFrame(hasubanan, smilesCol="canonical_smiles")
    hasubanan["taut_mol"] = hasubanan["ROMol"].apply(standardize_mol, canonicalize_tautomer=True)
    hasubanan["taut_smiles"] = hasubanan["taut_mol"].apply(Chem.MolToSmiles)

    return hasubanan
