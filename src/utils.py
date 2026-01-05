import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize


RDKIT_DESCRIPTORS = dict(Descriptors._descList)


# Simplified from https://www.blopig.com/blog/2022/05/molecular-standardization/
# stereo removal was added to comply with Axel's original procedure
def standardize_mol(mol, remove_stereo=False, canonicalize_tautomer=False):
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
            self, mol_col, filter_isotopes=True, min_heavy_atoms=3, max_heavy_atoms=75,
            filter_unsual_atoms=True, n_jobs=6
    ):
        self.mol_col = mol_col
        self.filter_isotopes = filter_isotopes
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.filter_unusual_atoms = filter_unsual_atoms
        self.n_jobs = n_jobs

    @property
    def medchem_atoms(self):
        return list(self._medchem_atoms.keys())

    @medchem_atoms.setter
    def medchem_atoms(self, value):
        periodic_table = Chem.GetPeriodicTable()
        elements = {}
        if isinstance(value, list):
            for v in value:
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
                f"Several chemical elements expected but only {value} given"
            )

    def _get_atom_set(self, mol):
        return {at.GetAtomicNum() for at in mol.GetAtoms()}

    def _has_non_medchem_atoms(self, mol):
        return len(self._get_atom_set(mol) - set(self._medchem_atoms.values())) > 0

    def _has_isotope(self, mol):
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    def identify_isotopes(self, df):
        df = df.copy()
        if self.n_jobs > 1:
            df["HasIsotopes"] = df[self.mol_col].parallel_apply(self._has_isotope)
        else:
            df["HasIsotopes"] = df[self.mol_col].apply(self._has_isotope)
        return df

    def identify_unusual_atoms(self, df):
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

    def get_heavy_atom_count(self, df):
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

    def drop_duplicates(self, df):
        df = df.copy()
        if self.n_jobs > 1:
            df["InchiKey"] = df[self.mol_col].parallel_apply(Chem.MolToInchiKey)
        else:
            df["InchiKey"] = df[self.mol_col].apply(Chem.MolToInchiKey)
        df.drop_duplicates(subset="InchiKey", inplace=True)
        return df

    def _drop_columns(self, df):
        cols_to_delete = [
            "HasIsotopes", "HasNonMedChemAtoms", "HeavyAtomCount", "InchiKey"
        ]
        existing_cols = set(cols_to_delete).intersection(df.columns)
        df.drop(columns=existing_cols, inplace=True)
        return df

    def filter(self, df):
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


def read_sdf(file):
    mol_supplier = Chem.MultithreadedSDMolSupplier(
        file, numWriterThreads=6, sanitize=False, removeHs=False
    )

    molecules = {}
    molecules["smiles"] = []
    first_mol = True
    for mol in mol_supplier:
        if mol is not None:
            # name = mol.GetProp("_Name")
            # molecules["Name"] = name
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
