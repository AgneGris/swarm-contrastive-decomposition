from pathlib import Path
from scd import train, save_results

# === Configuration ===
DATA_FILE = "emg.mat"
CONFIG_NAME = "surface"  # Just change this line for different experiments

# === Paths ===
HOME = Path.cwd() / "data"
input_path = HOME / "input" / DATA_FILE
output_path = HOME / "output" / f"{Path(DATA_FILE).stem}_{CONFIG_NAME}.pkl"
output_path.parent.mkdir(exist_ok=True)

# === Run ===
dictionary, timestamps = train(input_path, config_name=CONFIG_NAME)
save_results(output_path, dictionary)
print(f"Saved results to {output_path}")