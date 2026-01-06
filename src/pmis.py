"""
This module provides some functions necessary for the calculation of Principal Moments
of Inertia.

"""

import logging
from io import StringIO
from pathlib import PosixPath
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem, rdBase
from rdkit.Chem import PandasTools, rdDistGeom, rdForceFieldHelpers, rdMolDescriptors


rdBase.LogToPythonLogger()
logger = logging.getLogger("rdkit")
logger.handlers[0].setLevel(logging.WARN)
logger_sio = StringIO()
handler = logging.StreamHandler(logger_sio)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def get_message() -> str:
    """Retrieve RDKit logs."""
    text = logger_sio.getvalue()
    logger_sio.truncate(0)
    logger_sio.seek(0)
    if text:
        return text
    return ""


def generate_3d(mol: Chem.Mol, random_state: int) -> tuple[Optional[Chem.Mol], list]:
    """Check given molecule, generate and embedding and run force field-based
    minimization.

    Args:
        mol (Chem.Mol): molecule.
        random_state (int): random seed.

    Returns:
        tuple[Optional[Chem.Mol], list]: if minimization is successful, it returns the
                                         3D molecule together with a list of RDKit
                                         messages/warnings.
    """
    messages = []
    s = Chem.SanitizeMol(mol, catchErrors=True)
    messages.append(get_message())
    if s != 0:
        return None, messages
    mh = Chem.AddHs(mol)
    id = rdDistGeom.EmbedMolecule(mh, randomSeed=random_state)
    messages.append(get_message())
    if id == -1:
        # from https://github.com/rdkit/rdkit/issues/1433
        id = rdDistGeom.EmbedMolecule(mh, randomSeed=random_state, useRandomCoords=True)
        messages.append(get_message())
    if id == -1:
        return None, messages
    messages.append("")
    res = 10
    ntries = -1
    iters = [100, 300, 1000]
    while res > 0 and ntries < 3:
        res = rdForceFieldHelpers.UFFOptimizeMolecule(mh, maxIters=iters[ntries])
        ntries += 1
    messages.append(get_message())

    if res == 0:
        return mh, messages
    return None, messages


def calc_pmi(mol: Chem.Mol, replicates: int=3) -> tuple[float, float]:
    """Calculate normalized PMIs for the given molecule upon 2D to 3D transformation
    and minimization. The process is repeated `replicates` times to account for the
    random generation of conformers.

    Args:
        mol (Chem.Mol): molecule.
        replicates (int, optional): number of replicates to run. Defaults to 3.

    Returns:
        tuple[float, float]: median normalized PMIs for the molecule.
    """
    npr1 = []
    npr2 = []
    for rdst in [2025 + 3 * i for  i in range(replicates)]:
        m, msg = generate_3d(mol, rdst)  
        if m is not None:
            npr1.append(rdMolDescriptors.CalcNPR1(m))
            npr2.append(rdMolDescriptors.CalcNPR2(m))
        else:
            return np.nan, np.nan
    return np.median(npr1), np.median(npr2)


def get_pmis(filepath: Union[str, PosixPath]) -> pd.DataFrame:
    """Calculate normalized PMIs for a set of compounds contained in a file.

    Args:
        filepath (Union[str, PosixPath]): path to file containing SMILES.

    Returns:
        pd.DataFrame: normalized PMIs for all the compounds in the file.
                      Values are medians from repeated calculations.
    """
    df = pd.read_csv(filepath)
    smi_col = df.columns[df.columns.str.contains("smiles")][0]
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smi_col)
    if len(df) > 1000:
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        res = df["ROMol"].parallel_apply(calc_pmi)
        df[["NPR1", "NPR2"]] = pd.DataFrame(res.tolist(), index=df.index)
    else:
        df[["NPR1", "NPR2"]] = df["ROMol"].apply(calc_pmi).apply(pd.Series)
    cols = [df.columns[0]] + ["NPR1", "NPR2"]
    return df[cols]
