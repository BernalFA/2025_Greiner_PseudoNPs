import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit.Chem import Descriptors, PandasTools


RDKIT_DESCRIPTORS = dict(Descriptors._descList)


def get_descriptors_mol(mol):
    selected_descriptors = [
        "ExactMolWt",
        "RingCount",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "TPSA",
        "NumRotatableBonds",
        "fr_halogen",
        "NumBridgeheadAtoms",
        "FractionCSP3",
    ]
    calc_desc = {
        "HeavyAtoms": mol.GetNumAtoms()
    }
    for descriptor in selected_descriptors:
        function = RDKIT_DESCRIPTORS[descriptor]
        try:
            value = function(mol)
        except Exception:
            import traceback
            traceback.print_exc()
            value = np.nan
        calc_desc[descriptor] = value
    return calc_desc


def count_oxygen_atoms(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:
            count += 1
    return count


def count_nitrogen_atoms(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            count += 1
    return count


def count_lipinski_violations(desc):
    violations = [
        desc["ExactMolWt"] < 150 or desc["ExactMolWt"] > 500,
        desc["MolLogP"] > 5,
        desc["NumHDonors"] > 5,
        desc["NumHAcceptors"] > 10
    ]
    return sum(violations)


def count_veber_violations(desc):
    violations = [
        desc["NumRotatableBonds"] > 10,
        desc["TPSA"] > 140
    ]
    return sum(violations)


def calculate_selected_descriptors(mol):
    # calculate molecular descriptors
    descriptors = get_descriptors_mol(mol)
    # calculate atomic descriptors
    descriptors["NumOxygen"] = count_oxygen_atoms(mol)
    descriptors["NumNitrogen"] = count_nitrogen_atoms(mol)
    # calculate drug-like violations
    descriptors["LipinskiViolations"] = count_lipinski_violations(descriptors)
    descriptors["VeberViolations"] = count_veber_violations(descriptors)
    return descriptors


def get_descriptors_dataframe(filepath):
    df = pd.read_csv(filepath)
    try:
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol="taut_smiles")
    except KeyError:
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles")

    if len(df) > 1000:
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        descriptors = df["ROMol"].parallel_apply(
            calculate_selected_descriptors
        ).apply(pd.Series)
    else:
        descriptors = df["ROMol"].apply(
            calculate_selected_descriptors
        ).apply(pd.Series)
    result = pd.concat((df[df.columns[0]].copy(), descriptors), axis=1)
    return result
