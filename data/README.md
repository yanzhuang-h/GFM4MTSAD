# Data and pretrained models

Download MSL/SMAP from [Telemanom](https://github.com/khundman/telemanom),
and SMD from [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly).
Use the original MSL/SMAP arrays and OmniAnomaly-style preprocessed SMD arrays:

```text
<data_dir>/
├── SMAP_MSL/
│   ├── train/{P-15,T-3}.npy
│   └── test/{P-15,T-3}.npy, labeled_anomalies.csv
└── ServerMachineDataset/preprocessed/
    ├── machine-1-1_train.pkl
    ├── machine-1-1_test.pkl
    └── machine-1-1_test_label.pkl
```

Arrays are time × channels (MSL: 55, SMAP: 25, SMD: 38). Keep the original
label alignment; the loader fits normalization on the normal training split.

Download the registered AnomalyGFM source and pretrained weights:

```bash
mkdir -p assets/anomalygfm
curl -L --fail https://raw.githubusercontent.com/mala-lab/AnomalyGFM/4c9801a3867ceb0cca62c23e4591205b6584dceb/zero_shot/model.py \
  -o assets/anomalygfm/model.py
curl -L --fail https://raw.githubusercontent.com/mala-lab/AnomalyGFM/main/few_shot/model_weights_abnormal300.pth \
  -o assets/anomalygfm/model_weights_abnormal300.pth
cp configs/assets.example.json configs/assets.local.json
```

For OwlEye, obtain `model.py` from the [upstream repository](https://github.com/zhenglecheng/ICLR-2026-OWLEYE).
We pretrained its checkpoint, including the four pattern dictionaries, using
the paper and source code. **The matching checkpoint download is pending**;
the upstream link is a source-code reference, not the required weight download.

Set these paths in `configs/assets.local.json` (relative paths resolve beside that file):

| Key | Path |
|---|---|
| `data_dir` | Dataset root above |
| `anomalygfm_source` | Downloaded AnomalyGFM `model.py` |
| `anomalygfm_checkpoint` | Downloaded `model_weights_abnormal300.pth` |
| `owleye_source` | OwlEye `model.py` |
| `owleye_checkpoint` | Our OwlEye checkpoint and pattern dictionaries |

Only the selected backend's model files are required. The entrypoint verifies
their registered hashes before training. Check setup without starting a run:

```bash
python main.py check --entity SMD_1-1 --backend anomalygfm --assets configs/assets.local.json
```
