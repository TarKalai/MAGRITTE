"""Entry point: leakage-free GINE / LSTM / Transformer comparison on the BPIC 2012
signature-fraud benchmark. Thin runner -- the implementation lives in src/.

Reports threshold-independent metrics (ROC-AUC, PR-AUC) as mean +/- std over R repeated runs.

Usage: uv run python scripts/compare_models.py          # paper setting (R, epochs from config)
       uv run python scripts/compare_models.py 1 3      # quick smoke: R=1, 3 epochs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import pandas as pd

from config.experiment import CONFIG
from src.data.loaders import load_and_prepare
from src.methods.nested_cv import repeated_comparison


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else CONFIG["run"]["repeats"]
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else CONFIG["run"]["n_epochs"]
    data_path = CONFIG["data"]["path"]

    data = load_and_prepare(data_path)
    print(f"Comparison on {data_path}: R={R} repeats, {n_epochs} epochs, threshold-free metrics\n")
    table = repeated_comparison(
        *data,
        active_feature_columns=CONFIG["data"]["active_features"],
        R=R, n_epochs=n_epochs,
        model_kwargs=CONFIG["model"],
        **CONFIG["cv"],
    )
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
