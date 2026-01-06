"""
This script runs Principal Component Analysis (PCA) on the descriptors obtained after
running `03_feature_calculation.py`. Both PCA loadings and explained variance are saved
to disk.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


if __name__ == "__main__":
    HERE = Path.cwd()
    # Read descriptors file
    filepath = HERE / "reports" / "descriptors.csv"
    descriptors = pd.read_csv(filepath)
    feature_cols = [col for col in descriptors.columns if col not in ["ID", "dataset"]]
    # Define data
    X = descriptors.loc[:, feature_cols]
    # Scale
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    # Run PCA
    pca = PCA(n_components=10, random_state=2025)
    X_pca = pca.fit_transform(X_sc)
    # Store data
    X_pca_df = pd.DataFrame(X_pca)
    X_pca_df = pd.concat((descriptors.loc[:, ["ID", "dataset"]], X_pca_df), axis=1)
    X_pca_df.to_csv(HERE / "reports" / "PCA_scores.csv", index=False)
    pca.explained_variance_ratio_.tofile(HERE / "reports" / "PCA_exp_var.csv", sep=",")

    # Combine descriptors for alkaloid families
    comparisons = descriptors.query("dataset in ['pseudoNPs', 'Sceletium', 'Hasubanan']").copy()
    comparisons["new_set"] = comparisons["dataset"]
    for file in ["Amaryllidaceae.csv", "MIAs.csv"]:
        idx = pd.read_csv(HERE / "data" / file)
        tmp = descriptors.query("ID in @idx['chembl_id']").copy()
        tmp["new_set"] = file.split(".")[0]
        comparisons = pd.concat((comparisons, tmp), axis=0)
    # Repeat analysis on alkaloids only
    feature_cols_alk = [
        col for col in comparisons.columns if col not in ["ID", "dataset", "new_set"]
    ]
    X_alk = comparisons.loc[:, feature_cols_alk]

    scaler = StandardScaler()
    X_sc_alk = scaler.fit_transform(X_alk)

    pca = PCA(n_components=10, random_state=2025)
    X_pca_alk = pca.fit_transform(X_sc_alk)

    comp = comparisons.copy()
    comp.reset_index(inplace=True, drop=True)

    X_pca_alk_df = pd.DataFrame(X_pca_alk)
    X_pca_alk_df = pd.concat((comp.loc[:, ["ID", "new_set"]], X_pca_alk_df), axis=1)
    X_pca_alk_df.to_csv(HERE / "reports" / "PCA_scores_alkaloids.csv", index=False)
    pca.explained_variance_ratio_.tofile(
        HERE / "reports" / "PCA_exp_var_alkaloids.csv", sep=","
    )
