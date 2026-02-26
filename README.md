# PerforatedAI Dendrites — Heart Disease (Kaggle Playground S6E2) — Colab Notebook

This notebook trains a **tabular neural network** enhanced with **PerforatedAI dendrites** to predict the probability of **Heart Disease** for each `id` in `test.csv`, and generates a Kaggle-ready `submission.csv`.

---

## What it does

- Loads `train.csv` and `test.csv`
- Preprocesses:
  - **Numerical features** → `StandardScaler`
  - **Categorical features** → integer encoding + **embedding layers**
- Builds a fast tabular NN (MLP + BatchNorm + Dropout + Embeddings)
- Adds dendritic optimization via PerforatedAI:
  - `model = UPA.initialize_pai(model)`
  - Uses the PerforatedAI training loop:
    - train → validate AUC → `GPA.pai_tracker.add_validation_score(...)`
    - handles `restructured` and `training_complete`
- Produces `submission.csv` in required Kaggle format:

```csv
id,Heart Disease
630000,0.939262
630001,0.002786
...
```

---

## Requirements

### Colab runtime
- **Hardware accelerator:** GPU (recommended)
- Typical GPU: **NVIDIA Tesla T4**
- Works with CPU too, but slower.

### Python packages
Installed inside the notebook:
- `PerforatedAI` (from GitHub)
- `numpy`, `pandas`, `scikit-learn`
- Uses PyTorch (already available on Colab by default)

---

## Input files

Place these files in the Colab working directory (`/content/`):
- `train.csv`
- `test.csv`

Expected columns:
- `train.csv` contains: `id`, features, and target column `Heart Disease` with values `Presence` / `Absence`
- `test.csv` contains: `id` and the same feature columns (no target)

---

## How to run (Colab)

### 1) Enable GPU
Go to:
- **Runtime → Change runtime type → Hardware accelerator → GPU**

(Optional) Select **High-RAM** if available.

### 2) Verify GPU
The notebook prints CUDA status and GPU name, e.g. `Tesla T4`.

### 3) Run the notebook cells in order
The main training cell will:
- train the PerforatedAI dendrite model
- generate the output file:
  - `/content/submission.csv`

### 4) Download the submission
Use:

```python
from google.colab import files
files.download("/content/submission.csv")
```

(Optional) If you create a blended submission:
```python
files.download("/content/submission_blend.csv")
```

---

## Output artifacts

- `submission.csv` — primary submission file
- (Optional) `submission_blend.csv` — blended submission for better leaderboard stability

---

## Colab resources used (typical)

> Actual usage varies with batch sizes and configuration.

- **GPU:** Tesla T4 (recommended)
- **VRAM:** usually ~2–4 GB (tabular model is lightweight)
- **RAM:** a few GB for arrays + dataloaders
- **Runtime:**  
  - Fast run: ~6–7 minutes per seed (varies with Colab load and PAI restructuring)

---

## Notes about PerforatedAI prompts

PerforatedAI may warn that some modules (e.g., `BatchNorm1d`, `Embedding`) are not wrapped.
The notebook sets:
- `GPA.pc.set_unwrapped_modules_confirmed(True)`
- tracks module types to avoid `ipdb` interruptions.

PerforatedAI also recommends **no weight decay** for its training regime.
The notebook uses `weight_decay=0.0` by default.

---

## Troubleshooting

### 1) Submission save path error
- **Colab:** write to `/content/`
- **Kaggle:** write to `/kaggle/working/` (NOT `/kaggle/input/`)

### 2) Low leaderboard score (e.g., ~0.2–0.4)
Usually indicates a submission formatting / alignment issue:
- Header must be **exactly:** `id,Heart Disease`
- `id` order must match `test.csv`

Quick check:
```python
import numpy as np, pandas as pd
sub = pd.read_csv("submission.csv")
test = pd.read_csv("test.csv")
print(np.array_equal(sub["id"].values, test["id"].values))  # should be True
print(sub["Heart Disease"].min(), sub["Heart Disease"].max())
```

### 3) Runtime too long
Reduce:
- `MAX_OUTER`
- `TRAIN_STEPS_PER_OUTER`

Or enable a plateau-based early stop.

---

## Credits / Libraries
- **PerforatedAI** — dendrite optimization + model restructuring
- **PyTorch** — model definition + GPU training
- **scikit-learn** — scaling + ROC-AUC evaluation
- **pandas/numpy** — data loading and preprocessing
