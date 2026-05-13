import json
import glob

notebooks = [
    "All.ipynb",
    "FWI_Calculation.ipynb",
    "Hyperparameter_Tuning.ipynb",
    "Nearest_fire.ipynb",
    "Prediction_deep_learning.ipynb",
    "plotting.ipynb",
    "Data_Preparing/Modis_single_File.ipynb",
    "Data_Preparing/Trimming_UK.ipynb"
]

with open("summary.py", "w", encoding="utf-8") as out:
    for f in notebooks:
        try:
            with open(f, "r", encoding="utf-8") as nb_file:
                nb = json.load(nb_file)
            out.write(f"\n\n# ====== {f} ======\n\n")
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    out.write(source + "\n")
        except Exception as e:
            out.write(f"# Error reading {f}: {e}\n")
