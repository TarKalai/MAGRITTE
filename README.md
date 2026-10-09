# Graph- vs. Sequence-Based Fraud Detection on Process Logs

Comparison of a **GINE** graph neural network against **LSTM** and **Transformer** sequence
baselines for detecting fraudulent behaviour in event logs, evaluated under a **nested cross-validation** protocol with **threshold-independent** metrics.

Because real remote-desktop fraud data is private, the public benchmark injects a synthetic behavioural signature into the public **BPI Challenge 2012** log: a rare consecutive
transition (a back-loop `a → b → a` between two *common* activities) placed in the prefix
before the first transaction. Because both endpoints are common activities, the *presence* of
either is uninformative, only the abnormal *transition / ordering* is discriminative, which is
what makes the task a meaningful test of order-aware models. A fraction of legitimate traces
carry the same motif (class overlap) so the task stays non-trivial.

## Repository layout

Organised after the [VerbekeLab project template](https://github.com/VerbekeLab/template):

```
├── config/      experiment configuration (config/experiment.py)
├── data/
│   ├── raw/         original BPIC .xes logs (not committed — download separately)
│   └── processed/   bpic_2012_signature.parquet (the shipped benchmark)
├── lib/         off-the-shelf code (git-ignored); place the april subset here to regenerate data
├── notebooks/   per-model analysis + the dataset-generation notebook
├── res/         results / outputs
├── scripts/     executable entry points
├── src/
│   ├── data/        loaders.py  — log loading, chain-graph construction, features
│   ├── methods/     models.py (GINE/LSTM/Transformer), nested_cv.py (evaluation)
│   └── utils/       training.py — train/predict helpers
├── requirements.txt
└── README.md
```

| Key file | Purpose |
|----------|---------|
| `scripts/compare_models.py` | Thin entry point: comparison of GINE / LSTM / Transformer; reports **ROC-AUC** & **PR-AUC** as mean ± std over R runs. |
| `src/methods/models.py` | The three model definitions. |
| `src/methods/nested_cv.py` | Nested CV + repeated comparison. |
| `src/data/loaders.py` | Log loading, chain-graph construction, node/edge features, `load_and_prepare`. |
| `config/experiment.py` | Hyperparameters (data path, CV settings, model dims). |
| `scripts/regenerate_signature_dataset.py` | Canonical generator of the benchmark dataset. |
| `notebooks/Signature Fraud - BPIC 2012.ipynb` | Documented, step-by-step version of the generator. |
| `notebooks/main_{gine,lstm,transformer}_bpic2012.ipynb` | Per-model analysis (incl. progressive-prefix / earliness). |
| `data/processed/bpic_2012_signature.parquet` | The generated benchmark (shipped so results reproduce without the raw XES). |

> `lib/` is **git-ignored** (off-the-shelf code is not redistributed). It is only needed for the
> *optional* dataset regeneration — see step 1 below.

The raw BPIC `.xes` logs are **not** committed (they are hundreds of MB and publicly available — see below).

## Setup

Requires Python ≥ 3.12. The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # installs pm4py, pyarrow, ipykernel (+ numpy, pandas, scikit-learn, networkx)
```

PyTorch and PyTorch Geometric are **not** part of the locked dependencies (the CPU build is used
here) and must be installed explicitly:

```bash
uv pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch_geometric==2.8.0
```

> For GPU, install the matching CUDA build of `torch` from the PyTorch index instead of the `cpu` one.
> Reference versions used for the results below: `torch 2.12.0+cpu`, `torch_geometric 2.8.0`,
> `scikit-learn 1.9.0`, `networkx 3.6.1`, `pandas 3.0.3`, `numpy 2.4.6`.

## Reproduce the results

The benchmark parquet is already included, so you can skip straight to running the comparison.

**1. (Optional) Regenerate the benchmark dataset.** This step needs two things that are **not**
shipped with the repo:
- the raw log: download `BPI_Challenge_2012.xes` from [4TU.ResearchData](https://data.4tu.nl/)
  (search "BPI Challenge 2012") and place it at `data/raw/BPI_Challenge_2012.xes`;
- the [april](https://github.com/tnolle/binet) library's `processmining` subset (only
  `EventLog`/`Event`), placed at `lib/april/` (the loader expects `lib/april/processmining/...`).

Then:

```bash
uv run python scripts/regenerate_signature_dataset.py
# -> writes data/processed/bpic_2012_signature.parquet
```

**2. Run the model comparison** (`R` = number of repeated runs, then number of epochs):

```bash
uv run python scripts/compare_models.py            # defaults from config/experiment.py (R=3, 20 epochs)
uv run python scripts/compare_models.py 10 50      # 10 repeats, 50-epoch cap
```

**3. Per-model notebooks** — open `notebooks/main_{gine,lstm,transformer}_bpic2012.ipynb` and run all
cells (each chdir's to the repo root and loads `data/processed/bpic_2012_signature.parquet`).

## Method notes

- **Nested CV.** Outer `StratifiedGroupKFold` folds are scored once each; an inner
  group split provides the validation set used for early stopping (model selection by validation
  PR-AUC). Splits are grouped by session, so train / validation / test are disjoint and neither
  model selection nor any threshold ever sees the test fold.
- **Threshold-independent metrics only.** We report **ROC-AUC** and **PR-AUC** (average precision),
  which do not depend on a decision threshold, so the ranking is not an artefact of operating-point
  tuning.
- **Run-to-run variance.** Training is stochastic and runs multi-threaded on CPU, so exact numbers
  vary slightly between runs; report **mean ± std over `R` repeats** rather than a single run.

## Example results (BPI Challenge 2012 signature benchmark)

Nested CV, threshold-free metrics, mean ± std over repeated runs:

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| GINE | 0.880 ± 0.009 | 0.219 ± 0.007 |
| LSTM | 0.852 ± 0.011 | 0.210 ± 0.010 |
| Transformer | 0.875 ± 0.009 | 0.216 ± 0.013 |

The three models are close.



## License

The `april` library (needed only for the optional dataset regeneration) is not redistributed;
obtain it from https://github.com/tnolle/binet under its own license.

For the rest of the work: 
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
