"""
This module provides useful functions for the calculation of different molecular
descriptors and some typical drug-like violation counts.

"""

from pathlib import PosixPath
from typing import Union

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools


RDKIT_DESCRIPTORS = dict(Descriptors._descList)


def get_descriptors_mol(mol: Chem.Mol) -> dict:
    """Calculate a selected set of molecular descriptors for the given molecule. The
    descriptors set includes: 
        ExactMolWt
        RingCount
        NumAromaticRings
        NumAliphaticRings
        NumHDonors
        NumHAcceptors
        MolLogP
        TPSA
        NumRotatableBonds
        fr_halogen
        NumBridgeheadAtoms
        FractionCSP3

    Args:
        mol (Chem.Mol): molecule.

    Returns:
        dict: calculated descriptors.
    """
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


def count_oxygen_atoms(mol: Chem.Mol) -> int:
    """Calculate the number of Oxygen atoms in the molecule."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:
            count += 1
    return count


def count_nitrogen_atoms(mol: Chem.Mol) -> int:
    """Calculate the number of Nitrogen atoms in the molecule."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            count += 1
    return count


def count_lipinski_violations(desc: dict) -> int:
    """Check the number of violations to the Lipinski's rule of five.

    Args:
        desc (dict): descriptors calculated with `get_descriptors_mol`.

    Returns:
        int: number of violations.
    """
    violations = [
        desc["ExactMolWt"] < 150 or desc["ExactMolWt"] > 500,
        desc["MolLogP"] > 5,
        desc["NumHDonors"] > 5,
        desc["NumHAcceptors"] > 10
    ]
    return sum(violations)


def count_veber_violations(desc: dict) -> int:
    """Check the number of violations to the Veber's rules, where the number of
    rotatable bonds should not exceed 10 and TPSA should not exceed 140.

    Args:
        desc (dict): descriptors calculated with `get_descriptors_mol`.

    Returns:
        int: number of violations.
    """
    violations = [
        desc["NumRotatableBonds"] > 10,
        desc["TPSA"] > 140
    ]
    return sum(violations)


def calculate_selected_descriptors(mol: Chem.Mol) -> dict:
    """Calculate a set of molecular descriptors and drug-like violation counts for the
    given molecule.

    Args:
        mol (Chem.Mol): molecule.

    Returns:
        dict: calculated descriptors.
    """
    # calculate molecular descriptors
    descriptors = get_descriptors_mol(mol)
    descriptors["NumOxygen"] = count_oxygen_atoms(mol)
    descriptors["NumNitrogen"] = count_nitrogen_atoms(mol)
    # calculate drug-like violations
    descriptors["LipinskiViolations"] = count_lipinski_violations(descriptors)
    descriptors["VeberViolations"] = count_veber_violations(descriptors)
    return descriptors


def get_descriptors_dataframe(filepath: Union[str, PosixPath]) -> pd.DataFrame:
    """Calculate molecular descriptors and drug-like violation counts for all the
    molecules in a given compound collection.

    Args:
        filepath (Union[str, PosixPath]): path to file containing SMILES.

    Returns:
        pd.DataFrame: calculated descriptors for all the compounds in the file.
    """
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
