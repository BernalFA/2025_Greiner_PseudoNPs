"""
This module provides a helper function to simultaneously calculate NP likeness,
Quantitative Estimate of Drug-likeness (QED), Böttcher score, and the normalized
Spacial Score (SPS).

"""

import sys
from pathlib import Path, PosixPath
from typing import Union

import pandas as pd
from pandarallel import pandarallel
from rdkit.Chem import PandasTools, QED, SpacialScore

# Add path to RDKit contributions
# taken from https://github.com/rdkit/rdkit/issues/2279
from rdkit.Chem import RDConfig

sys.path.append(RDConfig.RDContribDir)

from NP_Score import npscorer

# The Böttcher score was calculated using the implementation from the Forli group.
# A clone of the original [repo](https://github.com/forlilab/bottchscore/tree/master)
# was used for it.
HERE = Path.cwd()
bottcher_path = HERE.parent.parent / "Downloads" / "bottchscore"

sys.path.append(str(bottcher_path))

from bottchscore3 import calculate_bottchscore_from_smiles

fscore = npscorer.readNPModel()
def score_np(mol):
    return npscorer.scoreMol(mol, fscore)


def get_scores(filepath: Union[str, PosixPath]) -> pd.DataFrame:
    """Calculate NP likeness, Quantitative Estimate of Drug-likeness (QED), Böttcher
    score, and the normalized Spacial Score (SPS) for all the compounds in the given
    file.

    Args:
        filepath (Union[str, PosixPath]): path to file containing SMILES.

    Returns:
        pd.DataFrame: calculated scores.
    """
    df = pd.read_csv(filepath)
    smi_col = df.columns[df.columns.str.contains("smiles")][0]
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smi_col)

    if len(df) > 1000:
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        df["NP_likeness"] = df["ROMol"].parallel_apply(score_np)
        df["QED"] = df["ROMol"].parallel_apply(QED.default)
        df["Boettcher"] = df[smi_col].parallel_apply(calculate_bottchscore_from_smiles)
        df["nSPS"] = df["ROMol"].parallel_apply(SpacialScore.SPS)
    else:
        df["NP_likeness"] = df["ROMol"].apply(score_np)
        df["QED"] = df["ROMol"].apply(QED.default)
        df["Boettcher"] = df[smi_col].apply(calculate_bottchscore_from_smiles)
        df["nSPS"] = df["ROMol"].apply(SpacialScore.SPS)
    cols = [df.columns[0]] + ["NP_likeness", "QED", "Boettcher", "nSPS"]
    return df[cols]
