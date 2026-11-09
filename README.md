# GFM4MTSAD

Code for **Graph Foundation Models Can Become Effective Multivariate Time Series Anomaly Detectors**.
Backends: AnomalyGFM (A) and OwlEye (O).

## Setup

Tested with Python 3.10, PyTorch 2.7.1, CUDA 12.8 and PyG 2.3.1.

```bash
git clone https://github.com/yanzhuang-h/GFM4MTSAD.git
cd GFM4MTSAD
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Follow [data and pretrained-model preparation](data/README.md), then set the paths in
`configs/assets.local.json`. The OwlEye checkpoint download is pending.

## Training and evaluation

```bash
DEVICE=0 ENTITY=SMD_1-1 BACKEND=anomalygfm SEED=0 \
OUTPUT=outputs/SMD_1-1_anomalygfm_seed0 bash scripts/run.sh
```

Use `MSL_P-15`, `SMAP_T-3` or `SMD_1-1`, with `anomalygfm` or `owleye`.
Each run pretrains its DLinear frontend for 30 epochs, then jointly trains the
frontend and adaptation modules for 30 epochs with the GFM backbone frozen.
Batch size is 256; checkpoints use normal/synthetic validation loss every five epochs.
Results and checkpoints are saved in `OUTPUT/frontend/` and `OUTPUT/joint/`.

To evaluate a saved joint checkpoint and its fixed normal references:

```bash
python main.py evaluate --entity SMD_1-1 --backend anomalygfm \
  --assets configs/assets.local.json --run-dir outputs/SMD_1-1_anomalygfm_seed0/joint \
  --output outputs/SMD_1-1_anomalygfm_seed0_eval
```

The reported metrics are AUROC, trapezoidal AUPRC and oracle F1 (test-label
threshold, no point adjustment), using window-any labels at length 60 and stride 1.
OwlEye uses tail consensus with tail mass 0.05 and fusion weight 0.5;
AnomalyGFM uses the unified node-to-time score. See `python main.py --help` for separate stages.

## License and acknowledgments

[Non-commercial license](LICENSE); commercial use requires permission.
We build on [CAROTS](https://github.com/kimanki/CAROTS),
[DLinear](https://github.com/cure-lab/LTSF-Linear),
[AnomalyGFM](https://github.com/mala-lab/AnomalyGFM) and
[OwlEye](https://github.com/zhenglecheng/ICLR-2026-OWLEYE).
Original licenses and attributions are retained in [licenses/](licenses/NOTICE).
