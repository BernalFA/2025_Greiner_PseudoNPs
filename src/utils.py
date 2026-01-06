"""
This module provides some utilities for molecule standardization, compound library
filtering, and file reading.

"""

from pathlib import PosixPath

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


RDKIT_DESCRIPTORS = dict(Descriptors._descList)


# Simplified from https://www.blopig.com/blog/2022/05/molecular-standardization/
# stereo removal was added to comply with Axel's original procedure
def standardize_mol(
        mol: Chem.Mol, remove_stereo: bool=False, canonicalize_tautomer: bool=False
    ) -> Chem.Mol:
    """Standardize the RDKit molecule, select its parent molecule, uncharge it,
    then enumerate all the tautomers.
    """
    # Follows the steps from:
    #  https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg Landrum) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ -- thanks JP!

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    mol = rdMolStandardize.Cleanup(mol)
    # if many fragments, get the "parent" (the actual mol we are interested in)
    mol = rdMolStandardize.FragmentParent(mol)
    # try to neutralize molecule
    # annoying, but necessary as no convenience method exists
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    # Note: no attempt is made at reionization at this step
    # nor ionization at some pH (RDKit has no pKa caculator);
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogues, etc.
    if remove_stereo:
        mol = rdMolStandardize.StereoParent(mol)
    if canonicalize_tautomer:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(100)
        mol = te.Canonicalize(mol)
    assert mol is not None
    return mol


class CompoundLibraryFilter:
    """Handles filtering of compound libraries. Compounds with unusual atoms (MedChem),
    unusual isotopes, large molecular weights (high heavy atom count), and duplicates
    are removed. 
    Atoms considered usual for MedChem can be changed with the medchem_atoms property.
    """
    _medchem_atoms = {
        "H": 1,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Br": 35,
        "I": 53,
    }

    def __init__(
            self, mol_col: str, filter_isotopes: bool=True, min_heavy_atoms: int=3,
            max_heavy_atoms: int=75, filter_unsual_atoms: bool=True, n_jobs: int=6
    ):
        """
        Args:
            mol_col (str): name of column containing molecule (RDKit Mol object).
            filter_isotopes (bool, optional): whether to filter off molecules with
                                              unusual isotopes. Defaults to True.
            min_heavy_atoms (int, optional): minimum number of heavy atoms allowed.
                                             Defaults to 3.
            max_heavy_atoms (int, optional): maximum number of heavy atoms allowed.
                                             Defaults to 75.
            filter_unsual_atoms (bool, optional): whether to filter off molecules with
                                                  unusual atoms. Defaults to True.
            n_jobs (int, optional): number of jobs to run in parallel. Defaults to 6.
        """
        self.mol_col = mol_col
        self.filter_isotopes = filter_isotopes
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.filter_unusual_atoms = filter_unsual_atoms
        self.n_jobs = n_jobs

    @property
    def medchem_atoms(self) -> list:
        """Get or set the chosen group of atoms to allow during filtering.
        
        A list of the chemical symbols for the allowed atoms is returned.

        Setting the medchem_atoms will create a completely new set of allowed atoms.
        For setting, a list of atomic numbers must be given.
        """
        return list(self._medchem_atoms.keys())

    @medchem_atoms.setter
    def medchem_atoms(self, values: list):
        periodic_table = Chem.GetPeriodicTable()
        elements = {}
        if isinstance(values, list):
            for v in values:
                try:
                    z = periodic_table.GetAtomicNumber(v)
                    if v not in elements:
                        elements[v] = z
                except RuntimeError:
                    raise ValueError(f"{v} is not a chemical element.")
            self._medchem_atoms = {
                k: v for k, v in sorted(elements.items(), key=lambda item: item[1])
            }
        else:
            raise ValueError(
                f"Several chemical elements expected but only {values} given"
            )

    def _get_atom_set(self, mol: Chem.Mol) -> set:
        """Retrieve atom types in the molecule."""
        return {at.GetAtomicNum() for at in mol.GetAtoms()}

    def _has_non_medchem_atoms(self, mol: Chem.Mol) -> int:
        """Check the number of unsual atoms."""
        return len(self._get_atom_set(mol) - set(self._medchem_atoms.values())) > 0

    def _has_isotope(self, mol: Chem.Mol) -> bool:
        """Check whether the molecule contains unusual or specified isotopes."""
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    def identify_isotopes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag molecules containing unusual or minor isotopes."""
        df = df.copy()
        if self.n_jobs > 1:
            df["HasIsotopes"] = df[self.mol_col].parallel_apply(self._has_isotope)
        else:
            df["HasIsotopes"] = df[self.mol_col].apply(self._has_isotope)
        return df

    def identify_unusual_atoms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag molecules containing unusual atoms."""
        df = df.copy()
        if self.n_jobs > 1:
            df["HasNonMedChemAtoms"] = df[self.mol_col].parallel_apply(
                self._has_non_medchem_atoms
            )
        else:
            df["HasNonMedChemAtoms"] = df[self.mol_col].apply(
                self._has_non_medchem_atoms
            )
        return df

    def get_heavy_atom_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the heavy atom count for all the molecules."""
        df = df.copy()
        if self.n_jobs > 1:
            df["HeavyAtomCount"] = df[self.mol_col].parallel_apply(
                RDKIT_DESCRIPTORS["HeavyAtomCount"]
            )
        else:
            df["HeavyAtomCount"] = df[self.mol_col].apply(
                RDKIT_DESCRIPTORS["HeavyAtomCount"]
            )
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eliminate duplicates based on Inchi Keys."""
        df = df.copy()
        if self.n_jobs > 1:
            df["InchiKey"] = df[self.mol_col].parallel_apply(Chem.MolToInchiKey)
        else:
            df["InchiKey"] = df[self.mol_col].apply(Chem.MolToInchiKey)
        df.drop_duplicates(subset="InchiKey", inplace=True)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe removing columns used for filtering."""
        cols_to_delete = [
            "HasIsotopes", "HasNonMedChemAtoms", "HeavyAtomCount", "InchiKey"
        ]
        existing_cols = set(cols_to_delete).intersection(df.columns)
        df.drop(columns=existing_cols, inplace=True)
        return df

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter off compounds with unusual/minor isotopes, unusual atoms (not common)
        in MedChem, outside the range of allowed heavy atoms, and duplicates.
        The resulting dataframe is a copy.

        Args:
            df (pd.DataFrame): compound collection.

        Returns:
            pd.DataFrame: filtered library.
        """
        if self.filter_isotopes:
            df = self.identify_isotopes(df)
            df = df.query("HasIsotopes == False").copy()
            print(f"After isotopes: {df.shape}")
        if self.filter_unusual_atoms:
            df = self.identify_unusual_atoms(df)
            df = df.query("HasNonMedChemAtoms == False").copy()
            print(f"After unusual: {df.shape}")

        df = self.get_heavy_atom_count(df)
        mask1 = (df["HeavyAtomCount"] >= self.min_heavy_atoms)
        mask2 = (df["HeavyAtomCount"] <= self.max_heavy_atoms)
        df = df[mask1 & mask2].copy()
        print(f"After heavy: {df.shape}")

        df = self.drop_duplicates(df)
        df = self._drop_columns(df)
        return df


def read_sdf(file: PosixPath) -> pd.DataFrame:
    """Utility to read SD files. All the properties in the SD file are stored as
    separate columns in the resulting dataframe.

    Args:
        file (PosixPath): path to file.

    Returns:
        pd.DataFrame: compound collection as dataframe.
    """
    mol_supplier = Chem.MultithreadedSDMolSupplier(
        file, numWriterThreads=6, sanitize=False, removeHs=False
    )

    molecules = {}
    molecules["smiles"] = []
    first_mol = True
    for mol in mol_supplier:
        if mol is not None:
            if first_mol:
                props = [prop for prop in mol.GetPropNames()]
                first_mol = False

            has_smiles = False
            for prop in props:
                if prop.lower() == "smiles":
                    has_smiles = True
                if mol.HasProp(prop):
                    if prop in molecules:
                        molecules[prop].append(mol.GetProp(prop))
                    else:
                        molecules[prop] = [mol.GetProp(prop)]
                else:
                    if prop in molecules:
                        molecules[prop].append("")
                    else:
                        molecules[prop] = [""]

            if not has_smiles:
                molecules["smiles"].append(Chem.MolToSmiles(mol))
        else:
            print("Failed mol")

    if len(molecules["smiles"]) == 0:
        molecules.pop("smiles", None)
    return pd.DataFrame(molecules)
