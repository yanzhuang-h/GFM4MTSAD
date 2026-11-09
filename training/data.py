from pathlib import Path
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.preprocessing import preprocess_splits

class NormalTrainingDataset(Dataset):
    """The same train split and normalization, opening no test arrays/labels."""

    def __init__(self, cfg, split="train"):
        name = str(cfg.DATA.NAME)
        base = Path(cfg.DATA.BASE_DIR)
        if name.startswith(("MSL_", "SMAP_")):
            self.source_path = base / "SMAP_MSL" / "train" / (name.split("_", 1)[1] + ".npy")
            raw = np.load(self.source_path, allow_pickle=False)
        elif name.startswith("SMD_"):
            self.source_path = base / "ServerMachineDataset" / "preprocessed" / ("machine-" + name.split("_", 1)[1] + "_train.pkl")
            with self.source_path.open("rb") as handle:
                raw = pickle.load(handle)
        else:
            raise ValueError("normal-only profiler supports MSL, SMAP, and SMD entities")
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != cfg.DATA.N_VAR or not np.isfinite(raw).all():
            raise ValueError("invalid normal training array")
        if cfg.DATA.PREPROCESS.PROTOCOL != "train_only" or not 0 < cfg.DATA.TRAIN_RATIO < 1:
            raise ValueError("requires train_only preprocessing and a normal train/validation split")
        if cfg.DATA.DOWNSAMPLE_RATE != 1:
            raise ValueError("profiler requires the registered undownsampled training data")
        boundary = int(len(raw) * cfg.DATA.TRAIN_RATIO)
        # Third argument is a discarded normal placeholder, never a test partition.
        self.train, self.val, _, self.preprocessing_diagnostics = preprocess_splits(
            raw[:boundary], raw[boundary:].copy(), raw[:1], scale=cfg.DATA.SCALE,
            protocol=cfg.DATA.PREPROCESS.PROTOCOL)
        self.train_labels = np.zeros(len(self.train))
        self.val_labels = np.zeros(len(self.val))
        if split not in ("train", "val"):
            raise ValueError("only normal train/val splits are supported")
        self.samples = self.train if split == "train" else self.val
        self.window = int(cfg.DATA.WIN_SIZE)
        self.stride = int(cfg.DATA.TRAIN_STEP if split == "train" else cfg.DATA.TEST_STEP)
        if len(self.samples) < self.window:
            raise ValueError("normal split is shorter than one window")

    def __len__(self):
        return (len(self.samples) - self.window) // self.stride + 1

    def __getitem__(self, index):
        start = index * self.stride
        return np.float32(self.samples[start:start + self.window]), 0



def make_loader(cfg, dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=cfg.TRAIN.SHUFFLE,
                      drop_last=cfg.TRAIN.DROP_LAST, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY)



def prepare_on_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=False)
    if isinstance(value, (tuple, list)):
        return type(value)(prepare_on_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: prepare_on_device(item, device) for key, item in value.items()}
    return value



def nested_digest(value):
    """Hash tensor contents and structure without altering state or RNG."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            digest.update(str((tuple(x.shape), x.dtype)).encode())
            digest.update(x.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for element in item:
                visit(element)
        elif isinstance(item, (str, int, float, bool, type(None))):
            digest.update(str((type(item).__name__, item)).encode())
        else:
            raise TypeError(f'unsupported integrity value: {type(item).__name__}')
    visit(value)
    return digest.hexdigest()
