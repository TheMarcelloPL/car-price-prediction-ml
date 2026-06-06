from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "sukhmandeepsinghbrar/car-price-prediction-dataset",
    output_dir=str(DATA_DIR),
)

print("Dane zapisano w:", path)