"""
This script helps calculate a set of 17 descriptors (molecular and drug-like features)
for all the compounds sets. Results are stored to file.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd()))

from src.features import get_descriptors_dataframe


if __name__ == "__main__":
    HERE = Path.cwd()
    path_data = HERE / "data" / "processed"

    files = [f for f in path_data.iterdir() if "cleaned" in f.name]

    descriptors = []
    for file in files:
        name = file.name.split("_")[0]
        df = get_descriptors_dataframe(file)
        df.insert(1, "dataset", [name] * len(df))
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)
        descriptors.append(df)

    descriptors = pd.concat(descriptors, axis=0)

    descriptors.to_csv(HERE / "reports" / "descriptors.csv",
                    index=False)
