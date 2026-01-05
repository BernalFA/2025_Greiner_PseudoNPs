import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd()))

from src.pmis import get_pmis


if __name__ == "__main__":
    HERE = Path.cwd()
    path_data = HERE / "data" / "processed"

    files = [f for f in path_data.iterdir() if "cleaned" in f.name]
    # Iterate over files and store results (time consuming due to number of compounds)
    for file in files:
        name = file.name.split("_")[0]
        df = get_pmis(file)
        df.insert(1, "dataset", [name] * len(df))
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)
        df.to_csv(HERE / "reports" / f"PMIs_{name}.csv", index=False)


    # Now combine results
    results_files = [f for f in (HERE / "reports").iterdir() if "PMIs_" in f.name]

    pmis = []
    for file in results_files:
        df = pd.read_csv(file)
        pmis.append(df)

    pmis = pd.concat(pmis, axis=0)
    pmis.to_csv(HERE / "reports" / "PMIs.csv", index=False)
