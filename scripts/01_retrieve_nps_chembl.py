import csv
import sqlite3
from pathlib import Path


# Define paths
HERE = Path.cwd().parent
LIBRARIES_PATH = HERE.parent / "compound_collections"
CHEMBL_PATH = LIBRARIES_PATH / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"

# Define SQL statement to retrieve ChEMBL ID and SMILES for compounds flagged
# as natural products
sql = """
SELECT DISTINCT md.chembl_id AS Chembl_Id,
                cs.canonical_smiles AS Canonical_Smiles

FROM molecule_dictionary md
JOIN compound_structures cs ON cs.molregno = md.molregno

WHERE md.natural_product IN ('1')
"""


if __name__ == "__main__":
    # access ChEMBL
    conn = sqlite3.connect(CHEMBL_PATH)
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    conn.close()
    print(f"Query resulted in {len(res)} compounds")

    # Store information as CSV
    with open(HERE / "data" / "raw" / "chembl_35_NP.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["chembl_id", "smiles"])
        for row in res:
            writer.writerow(row)
