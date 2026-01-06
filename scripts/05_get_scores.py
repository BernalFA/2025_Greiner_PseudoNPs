from pathlib import Path

import pandas as pd

# Add path to RDKit contributions
# taken from https://github.com/rdkit/rdkit/issues/2279
import sys

sys.path.append(str(Path.cwd()))

from src.scores import get_scores

if __name__ == "__main__":
    HERE = Path.cwd()
    path_data = HERE / "data" / "processed"
    files = [f for f in path_data.iterdir() if "cleaned" in f.name]
    files.append(HERE / "data" / "Sceletium.csv")

    scores = []
    for file in files:
        name = file.name.split("_")[0]
        if name == "pseudo":
            name = "pseudoNPs"
        df = get_scores(file)
        df.insert(1, "dataset", [name] * len(df))
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)
        scores.append(df)

    scores = pd.concat(scores, axis=0)

    scores.to_csv(HERE / "reports" / "scores.csv", index=False)
