# Third-party notices

| Component | Source | Handling |
|---|---|---|
| CAROTS training scaffold | https://github.com/kimanki/CAROTS | Original [non-commercial license with permission clause](licenses/CAROTS-LICENSE.txt) retained. |
| DLinear | https://github.com/cure-lab/LTSF-Linear | Original [Apache-2.0 license](licenses/DLinear-LICENSE.txt) retained. |
| AnomalyGFM | https://github.com/mala-lab/AnomalyGFM | Load upstream model source and pretrained weights externally; adaptation wrapper is included. |
| OwlEye | https://github.com/zhenglecheng/ICLR-2026-OWLEYE | Load upstream model source and our separately pretrained checkpoint externally. |

Datasets, third-party GFM source, and model weights are not bundled. Their
respective upstream terms apply; no new license is granted to those materials.
The AnomalyGFM loader omits an unused sparse-library import and applies the
validated adjacency device/dtype fix in memory, leaving the downloaded file
unchanged. These compatibility changes preserve parameter names and computations.
