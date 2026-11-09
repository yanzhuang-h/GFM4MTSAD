# Pretrained models

Download source and weights separately, then set their paths in
`configs/assets.local.json`. Only the selected backend's assets are required.
The example below prepares AnomalyGFM:

```bash
mkdir -p assets/anomalygfm
curl -L --fail https://raw.githubusercontent.com/mala-lab/AnomalyGFM/4c9801a3867ceb0cca62c23e4591205b6584dceb/zero_shot/model.py \
  -o assets/anomalygfm/model.py
curl -L --fail https://raw.githubusercontent.com/mala-lab/AnomalyGFM/main/few_shot/model_weights_abnormal300.pth \
  -o assets/anomalygfm/model_weights_abnormal300.pth
cp configs/assets.example.json configs/assets.local.json
```

For OwlEye, obtain `model.py` from the
[official repository](https://github.com/zhenglecheng/ICLR-2026-OWLEYE).
Our experiments use a checkpoint pretrained by us following that paper and
code, including all four learned pattern dictionaries. **Its download link is
pending.** The upstream repository link is a source-code reference, not a link
to the exact checkpoint required below. A matching checkpoint is required to
run the OwlEye backend.

| Asset JSON key | Expected content |
|---|---|
| `data_dir` | Root described in [data preparation](../data/README.md) |
| `anomalygfm_source` | Upstream `zero_shot/model.py` |
| `anomalygfm_checkpoint` | `model_weights_abnormal300.pth` |
| `owleye_source` | Upstream `model.py` |
| `owleye_checkpoint` | Our pretrained weights and learned dictionaries |

The training entrypoint checks the registered assets before starting:

```bash
python main.py check --entity SMD_1-1 --backend anomalygfm \
  --assets configs/assets.local.json
```

Expected SHA256 values:

```text
AnomalyGFM weights  b6783375a0c340acd2b5ee42bae40280b7a2adb5dc27a82b8ca0504ac8671413
AnomalyGFM source   870ff2d6cefef715a4fd701f001a9bda98dd7f808a3c431d5edec37e01e3ace1
OwlEye weights      fd35544a57e8e6ffa271ad0e68719fdec8648fd2a2d08c7db5de7f62ebc57fa2
OwlEye source       b1944d38298ce23163ab6f9d64a8607d46063bc7c9a5a0dd84285dde5746d319
```

AnomalyGFM source hashing normalizes line endings and final whitespace; weights
and OwlEye source use exact file bytes. GFM checkpoints are distinct from the
frontend and joint checkpoints trained by this repository. Source files and
weights remain local assets and are excluded from Git.
