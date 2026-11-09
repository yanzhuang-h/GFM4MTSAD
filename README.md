# GFM4MTSAD

Implementation of **Graph Foundation Models Can Become Effective Multivariate
Time Series Anomaly Detectors**, with AnomalyGFM (A) and OwlEye (O) backends.

## Installation

Tested with Python 3.10, PyTorch 2.7.1, CUDA 12.8, and PyTorch Geometric 2.3.1.

```bash
git clone https://github.com/yanzhuang-h/GFM4MTSAD.git
cd GFM4MTSAD
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data and pretrained models

Prepare [MSL, SMAP and SMD](data/README.md), then follow the
[pretrained model instructions](docs/ASSETS.md).
Copy `configs/assets.example.json` to `configs/assets.local.json` and set your
data, model-source and checkpoint paths. Training requires no network access.

Supported entities: `MSL_P-15`, `SMAP_T-3`, `SMD_1-1`.
Supported backends: `anomalygfm`, `owleye`.
The OwlEye checkpoint was pretrained by us using its paper and source code;
its download link will be added separately.

## Training

```bash
DEVICE=0 ENTITY=SMD_1-1 BACKEND=anomalygfm SEED=0 \
OUTPUT=outputs/SMD_1-1_anomalygfm_seed0 bash scripts/run.sh
```

This runs **30 epochs of frontend pretraining followed by 30 epochs of
warm-start joint training**, and evaluates the selected model. During joint
training, the temporal frontend and adaptation modules are trainable, while
the pretrained GFM backbone stays frozen. Checkpoints are selected every five
epochs using normal/synthetic validation loss.

Change `ENTITY`, `BACKEND`, `SEED`, and `OUTPUT` for another run. Results are
saved under `OUTPUT/frontend/` and `OUTPUT/joint/`; use a new output directory
for each run. Separate `pretrain` and `train` commands are also available through
`python main.py --help`.

## Evaluation

Reload a trained model and its saved normal references:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py evaluate \
  --entity SMD_1-1 --backend anomalygfm \
  --assets configs/assets.local.json \
  --run-dir outputs/SMD_1-1_anomalygfm_seed0/joint \
  --output outputs/SMD_1-1_anomalygfm_seed0_eval
```

AUROC, trapezoidal AUPRC and oracle F1 are saved with the score arrays.
See [the training and evaluation protocol](docs/PROTOCOL.md) for scoring details.

## Acknowledgments

Our implementation builds on [CAROTS](https://github.com/kimanki/CAROTS),
[DLinear](https://github.com/cure-lab/LTSF-Linear),
[AnomalyGFM](https://github.com/mala-lab/AnomalyGFM), and
[OwlEye](https://github.com/zhenglecheng/ICLR-2026-OWLEYE).
Upstream attribution and terms are retained in [third-party notices](docs/THIRD_PARTY.md).
Paper experiment scripts are reserved under `experiments/`.
