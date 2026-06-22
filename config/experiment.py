"""Experiment configuration for the BPIC 2012 signature-fraud model comparison.

Plain-Python config (no extra dependency). Values mirror the paper setting; the number of
repeats and epochs can be overridden on the command line:
    python scripts/compare_models.py [R] [N_EPOCHS]
"""
CONFIG = {
    "data": {
        "path": "data/processed/bpic_2012_signature.parquet",
        "active_features": ["cumulative_time", "event_frequency_rank"],
    },
    "run": {"repeats": 3, "n_epochs": 20},
    "cv": {
        "n_splits": 5, "n_inner_splits": 4, "n_seeds": 3, "patience": 5,
        "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 32, "cut_only_training": True,
    },
    "model": {"event_emb_dim": 16, "hidden_dim": 32, "n_layers": 2, "dropout": 0.0},
}
