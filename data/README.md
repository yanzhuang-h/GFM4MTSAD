# Data preparation

Use the original Telemanom MSL/SMAP arrays and anomaly annotations, and the
OmniAnomaly-style SMD preprocessed arrays. Obtain data from their upstream
projects; data archives are not bundled here.

- MSL / SMAP: https://github.com/khundman/telemanom
- SMD: https://github.com/NetManAIOps/OmniAnomaly

Point `data_dir` in the asset JSON to the directory containing:

```text
SMAP_MSL/
  train/P-15.npy
  train/T-3.npy
  test/P-15.npy
  test/T-3.npy
  test/labeled_anomalies.csv
ServerMachineDataset/preprocessed/
  machine-1-1_train.pkl
  machine-1-1_test.pkl
  machine-1-1_test_label.pkl
```

Arrays have shape time × channels: 55 for MSL_P-15, 25 for SMAP_T-3,
and 38 for SMD_1-1. Labels must align with the original test timeline.
Do not externally normalize these arrays. The pipeline fits preprocessing on
the normal training partition and applies it to validation/test data. Training
uses window length 60 and stride 1. The dataset loader records the window-label
and evaluation-start rules in the metric output.

Frontend and joint training use contiguous normal train/validation splits and
do not open test arrays through their training loaders. Test data are read by
final evaluation. Only load trusted pickle/checkpoint files from the expected
data and model sources.
