# Training and evaluation

- Window length 60, stride 1; the normal sequence is split chronologically into
  training and validation blocks. Preprocessing is fitted on training data only.
- Each entity/backend/seed gets independent frontend pretraining. Only its
  selected DLinear parameters initialize the subsequent joint stage; Bridge,
  IPRA, backend adaptation and readout are initialized for that joint stage.
- Both stages run for 30 epochs with batch size 256. Validation every five
  epochs uses held-out normal windows and synthetic views, without test labels.
- The pretrained GFM backbone remains frozen. Normal graph/evidence references
  follow the recipe's train-normal refresh rules and are saved for inference.
- AnomalyGFM reports the native unified node-to-time score. OwlEye additionally
  applies train-tail-gated consensus with tail mass 0.05 and fusion weight 0.5.
- AUROC and AUPRC use window scores; AUPRC is trapezoidal area, not AP.
  Reported F1 uses the oracle threshold selected on test labels, without point
  adjustment. It is not a deployed-threshold estimate.

The six recipes in `configs/registered/` override shared defaults in `config.py`.
The entrypoint then supplies paths, seed, and stage settings, saving the complete
effective configuration with each run. Relative asset paths are resolved beside
the asset JSON. Other names inside checkpoints retain the original `SHARK`
namespace for compatibility.

`main.py evaluate` reloads the selected model, saved calibration tensors and
normal mechanism. It checks the inference bundle and keeps the saved references
fixed. Original and reloaded results were checked across all six entity/backend
conditions after complete 30+30-epoch training.
