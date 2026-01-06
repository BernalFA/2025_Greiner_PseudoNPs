import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

sys.path.append(str(Path.cwd()))

from src.utils import read_sdf  # noqa: E402
from src.data_preparation import (
    prepare_chembl_with_sugars, prepare_chembl_no_sugars, prepare_amaryllidaceae,
    prepare_mias, prepare_hasubanan, prepare_sceletium, process_library
)

# disable RDKit C++ log
RDLogger.DisableLog('rdApp.warning')

# Define path
HERE = Path.cwd()
path_data = HERE / "data"
LIBRARIES_PATH = HERE.parent / "compound_collections"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        choices=[
                            "chembl_sugars",
                            "chembl_no_sugars",
                            "drugbank",
                            "enamine",
                            "pnps",
                            "mias",
                            "amaryllidaceae",
                            "sceletium",
                            "hasubanan"
                        ])
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "chembl_sugars":
        chembl = pd.read_csv(path_data / "raw" / "chembl_35_NP.csv")
        chembl.rename(columns={"chembl_id": "ID"}, inplace=True)
        chembl_filtered = prepare_chembl_with_sugars()
        chembl[["ID", "clean_smiles"]].to_csv(
            path_data / "interim" / "chembl_35_NP_for_deglycosilation.csv",
            index=False
        )
    elif args.dataset == "chembl_no_sugars":
        chembl_no_sugars_filtered = prepare_chembl_no_sugars()
        chembl_no_sugars_filtered[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "chembl_35_NP_cleaned.csv", index=False
        )
    elif args.dataset == "drugbank":
        file = LIBRARIES_PATH / "drugbank_5_1_13" / "drugbank_5_1_13_structures.sdf"
        drugbank = read_sdf(file)
        drugbank.rename(columns={"DATABASE_ID": "ID"}, inplace=True)
        drugbank_filtered = process_library(drugbank, smilesCol="SMILES", workers=48)
        drugbank_filtered[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "drugbank_5_1_13_cleaned.csv", index=False
        )
    elif args.dataset == "enamine":
        enamine_path = LIBRARIES_PATH / "enamine_screening_collection_202504"
        molecules = read_sdf(enamine_path / "Enamine_screening_collection_202504.sdf")
        advanced_enamine = molecules.query("Collection == 'Advanced'").copy()
        advanced_enamine.rename(columns={"Catalog_ID": "ID"}, inplace=True)
        enamine_subset_filtered = process_library(advanced_enamine, workers=48,
                                                subsample=True)
        enamine_subset_filtered[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "enamine_advanced_50k_subset_cleaned.csv",
            index=False
        )
    elif args.dataset == "pnps":
        path_file = path_data / "raw" / "pseudo-NPs.csv"
        compounds = pd.read_csv(path_file)
        compounds.columns = ["ID", "smiles"]
        compounds_filtered = process_library(compounds)
        compounds_filtered[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "compounds_GreinerL_cleaned.csv", index=False
        )
    elif args.dataset == "mias":
        chembl = pd.read_csv(path_data / "processed" / "chembl_35_NP_cleaned.csv")
        mias = prepare_mias(chembl)
        mias[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "MIAs.csv", index=False
        )
    elif args.dataset == "amaryllidaceae":
        chembl = pd.read_csv(path_data / "processed" / "chembl_35_NP_cleaned.csv")
        amaryllidaceae = prepare_amaryllidaceae(chembl)
        amaryllidaceae[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "Amaryllidaceae.csv", index=False
        )
    elif args.dataset == "sceletium":
        sceletium = prepare_sceletium()
        sceletium.to_csv(
            path_data / "interim" / "Sceletium_compounds.csv", index=False
        )
    elif args.dataset == "hasubanan":
        coconut = pd.read_csv(LIBRARIES_PATH / "coconut_08_25.csv")
        hasubanan = prepare_hasubanan(coconut)
        hasubanan[["ID", "taut_smiles"]].to_csv(
            path_data / "processed" / "Hasubanan_cleaned.csv", index=False
        )
    else:
        raise ValueError(f"{args.dataset} not recognized.")
