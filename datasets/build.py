# adapted from: https://github.com/thuml/Anomaly-Transformer/data_factory/data_loader.py
import ast
import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from datasets.preprocessing import preprocess_splits
from datasets.util import downsample, downsample_binary_labels


class TSDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.dataset_name = cfg.DATA.NAME
        self.data_dir = os.path.join(cfg.DATA.BASE_DIR, cfg.DATA.NAME)
        self.win_size = cfg.DATA.WIN_SIZE
        self.step = cfg.DATA.TRAIN_STEP if split == "train" else cfg.DATA.TEST_STEP
        self.scale = cfg.DATA.SCALE
        self.split = split
        self.train_ratio = cfg.DATA.TRAIN_RATIO
        self.downsample_rate = cfg.DATA.DOWNSAMPLE_RATE

        self.train, self.val, self.test, self.train_labels, self.val_labels, self.test_labels = self._load_data()
        self._validate_loaded_data()
        self.test_labels = np.squeeze(self.test_labels)
        self._normalize_data()

    def _validate_loaded_data(self):
        """在归一化前验证时间轴、变量数和标签长度，尽早暴露数据错位。"""
        arrays = {
            "train": (self.train, self.train_labels),
            "val": (self.val, self.val_labels),
            "test": (self.test, self.test_labels),
        }
        feature_counts = set()
        for split, (features, labels) in arrays.items():
            features = np.asarray(features)
            labels = np.asarray(labels)
            if features.ndim != 2:
                raise ValueError(
                    f"{self.dataset_name} {split} features must be two-dimensional, "
                    f"got {features.shape}"
                )
            if labels.ndim not in (1, 2):
                raise ValueError(
                    f"{self.dataset_name} {split} labels must be one- or two-dimensional, "
                    f"got {labels.shape}"
                )
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{self.dataset_name} {split} feature/label length mismatch: "
                    f"{features.shape[0]} != {labels.shape[0]}"
                )
            feature_counts.add(int(features.shape[1]))
        if len(feature_counts) != 1:
            raise ValueError(
                f"{self.dataset_name} feature dimensions disagree across splits: "
                f"{sorted(feature_counts)}"
            )
        expected = int(self.cfg.DATA.N_VAR)
        actual = next(iter(feature_counts))
        if actual != expected:
            raise ValueError(
                f"{self.dataset_name} feature dimension disagrees with DATA.N_VAR: "
                f"{actual} != {expected}"
            )

    def _downsampled_path(self, name):
        tag = getattr(self, "downsample_cache_tag", "")
        suffix = f"_{tag}" if tag else ""
        return os.path.join(
            self.data_dir,
            f"downsampled_{name}_rate{self.downsample_rate}{suffix}.npy",
        )

    def _save_downsampled_data(self, train, val, test, train_labels, val_labels, test_labels):
        assert self.downsample_rate > 1
        np.save(self._downsampled_path("train"), train)
        np.save(self._downsampled_path("val"), val)
        np.save(self._downsampled_path("test"), test)
        np.save(self._downsampled_path("train_labels"), train_labels)
        np.save(self._downsampled_path("val_labels"), val_labels)
        np.save(self._downsampled_path("test_labels"), test_labels)

    def _load_downsampled_data(self):
        train = np.load(self._downsampled_path("train"))
        val = np.load(self._downsampled_path("val"))
        test = np.load(self._downsampled_path("test"))
        train_labels = np.load(self._downsampled_path("train_labels"))
        val_labels = np.load(self._downsampled_path("val_labels"))
        test_labels = np.load(self._downsampled_path("test_labels"))
        return train, val, test, train_labels, val_labels, test_labels

    def _load_data(self):
        raise NotImplementedError

    def _split_train_val(self, train, train_labels):
        # use the latter portion of the training data as validation data.
        assert self.train_ratio < 1.0

        train_len = int(len(train) * self.train_ratio)
        val = train[train_len:].copy()
        val_labels = train_labels[train_len:].copy()
        train = train[:train_len]
        train_labels = train_labels[:train_len]

        return train, val, train_labels, val_labels

    def _normalize_data(self):
        preprocess = getattr(self.cfg.DATA, "PREPROCESS", None)
        protocol = getattr(preprocess, "PROTOCOL", "train_only")
        (
            self.train,
            self.val,
            self.test,
            self.preprocessing_diagnostics,
        ) = preprocess_splits(
            self.train,
            self.val,
            self.test,
            scale=self.scale,
            protocol=protocol,
        )

    def print_data_stats(self):
        if self.split == 'train':
            print(f"Train data shape: {self.train.shape}, mean: {np.mean(self.train, axis=0)}, std: {np.std(self.train, axis=0)}")
        elif self.split == 'val':
            print(f"Validation data shape: {self.val.shape}, mean: {np.mean(self.val, axis=0)}, std: {np.std(self.val, axis=0)}")
        elif self.split == 'test':
            print(f"Test data shape: {self.test.shape}, mean: {np.mean(self.test, axis=0)}, std: {np.std(self.test, axis=0)}")

    def __len__(self):
        if self.split == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.split == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.split == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.split == "train":
            inputs = self.train
            labels = self.train_labels
        elif self.split == 'val':
            inputs = self.val
            labels = self.val_labels
        elif self.split == 'test':
            inputs = self.test
            labels = self.test_labels

        return np.float32(inputs[index:index + self.win_size]), int(np.any(labels[index:index + self.win_size] == 1))


class PSMSegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(PSMSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        test_label_path = os.path.join(self.data_dir, 'test_label.csv')

        train = pd.read_csv(train_path)
        train = train.values[:, 1:]
        train = np.nan_to_num(train)
        # placeholder for train_labels
        train_labels = np.zeros(train.shape[0])

        test = pd.read_csv(test_path)
        test = test.values[:, 1:]
        test = np.nan_to_num(test)
        test_labels = pd.read_csv(test_label_path).values[:, 1:]

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class NASASegLoader(TSDataset):
    spacecraft = None

    def __init__(self, cfg, split):
        parts = str(cfg.DATA.NAME).split("_", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"NASA dataset name must include an entity, got {cfg.DATA.NAME}"
            )
        self.entity = parts[1]
        super(NASASegLoader, self).__init__(cfg, split)

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'SMAP_MSL')

        train = np.load(os.path.join(self.data_dir, "train", f"{self.entity}.npy"))
        test = np.load(os.path.join(self.data_dir, "test", f"{self.entity}.npy"))

        label_file = os.path.join(self.data_dir, 'test', 'labeled_anomalies.csv')
        df = pd.read_csv(label_file)
        rows = df.loc[df['chan_id'] == self.entity].drop_duplicates()
        if len(rows) != 1:
            raise ValueError(
                f"Expected one unique metadata row for {self.entity}, got {len(rows)}"
            )
        actual_spacecraft = str(rows['spacecraft'].item())
        if actual_spacecraft != self.spacecraft:
            raise ValueError(
                f"Entity {self.entity} belongs to {actual_spacecraft}, not {self.spacecraft}"
            )
        anomaly_sequences = ast.literal_eval(rows['anomaly_sequences'].item())

        test_labels = np.zeros(test.shape[0])
        for start, end in anomaly_sequences:
            test_labels[start: end + 1] = 1
        test_labels = test_labels.astype(int)

        train_labels = np.zeros(train.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class MSLSegLoader(NASASegLoader):
    spacecraft = "MSL"


class SMAPSegLoader(NASASegLoader):
    spacecraft = "SMAP"


class SMDSegLoader(TSDataset):
    def __init__(self, cfg, split):
        self.entity = cfg.DATA.NAME.split('_')[1]
        super(SMDSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'ServerMachineDataset')
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_test.pkl'), 'rb') as f:
            test = pickle.load(f)
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_test_label.pkl'), 'rb') as f:
            test_labels = pickle.load(f)
        test_labels = test_labels.astype(int)
        # placeholder for train, val labels
        train_labels = np.zeros(train.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class TSBADMSegLoader(TSDataset):
    """Loader for one independent multivariate series from TSB-AD-M.

    TSB-AD-M stores each series as a CSV whose final column is ``Label`` and
    whose filename declares the normal training prefix (``tr_<n>``).  The
    loader deliberately treats that prefix as the only training source and
    never uses labels from the test suffix for fitting or validation.
    """

    FILES = {
        "TSB_SWAT_1": "171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv",
    }

    def __init__(self, cfg, split):
        self.entity = str(cfg.DATA.NAME)
        if self.entity not in self.FILES:
            raise ValueError(
                f"Unsupported TSB-AD-M entity {self.entity}; "
                f"expected one of {sorted(self.FILES)}"
            )
        self.source_path = None
        self.train_prefix_length = None
        self.first_anomaly_index = None
        self.source_metadata = {}
        super().__init__(cfg, split)

    @classmethod
    def _candidate_roots(cls, cfg):
        """Yield deterministic roots, allowing a server-side data override."""
        configured = os.environ.get("TSB_AD_M_ROOT")
        if configured:
            yield Path(configured).expanduser()
        base = Path(str(cfg.DATA.BASE_DIR)).expanduser()
        yield base / "TSB-AD-M"
        yield base / "TSB_AD_M"
        repo_root = Path(__file__).resolve().parents[1]
        yield repo_root / "baseline" / "TSB-AD" / "Datasets" / "TSB-AD-M"
        yield repo_root / "data" / "TSB-AD-M"

    @classmethod
    def _find_source(cls, cfg, filename):
        checked = []
        for root in cls._candidate_roots(cfg):
            path = root / filename
            checked.append(str(path))
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(
            "TSB-AD-M source file not found. Set TSB_AD_M_ROOT or place the "
            f"file under data/TSB-AD-M. Checked: {', '.join(checked)}"
        )

    @staticmethod
    def _parse_filename_metadata(path):
        match = re.search(r"_tr_(\d+)_1st_(\d+)(?:\.[^.]+)?$", path.name)
        if match is None:
            raise ValueError(
                "TSB-AD-M filename must declare training prefix and first "
                f"anomaly (got {path.name!r})"
            )
        return int(match.group(1)), int(match.group(2))

    def _load_data(self):
        source = self._find_source(self.cfg, self.FILES[self.entity])
        train_prefix, first_anomaly = self._parse_filename_metadata(source)
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"TSB-AD-M source is empty: {source}")
        labels_name = str(frame.columns[-1]).strip()
        if labels_name.lower() != "label":
            raise ValueError(
                f"TSB-AD-M final column must be Label, got {labels_name!r}"
            )
        feature_frame = frame.iloc[:, :-1].apply(pd.to_numeric, errors="raise")
        labels_series = pd.to_numeric(frame.iloc[:, -1], errors="raise")
        features = feature_frame.to_numpy(dtype=np.float32)
        labels_numeric = labels_series.to_numpy(dtype=np.float64)
        if not np.isfinite(features).all() or not np.isfinite(labels_numeric).all():
            raise ValueError(f"TSB-AD-M source contains NaN/Inf: {source}")
        unique_labels = set(np.unique(labels_numeric).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"TSB-AD-M labels must be binary 0/1, got {sorted(unique_labels)}"
            )
        if not 0 < train_prefix < len(features):
            raise ValueError(
                f"TSB-AD-M training prefix {train_prefix} is outside source "
                f"length {len(features)}"
            )
        prefix_labels = labels_numeric[:train_prefix]
        if np.any(prefix_labels != 0):
            raise ValueError(
                "TSB-AD-M training prefix contains anomaly labels; refusing "
                "to leak labels into normal-only training"
            )

        self.data_dir = str(source.parent)
        self.source_path = str(source)
        self.train_prefix_length = int(train_prefix)
        self.first_anomaly_index = int(first_anomaly)
        self.var_names = [str(name) for name in feature_frame.columns]
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.source_metadata = {
            "filename": source.name,
            "path": str(source),
            "sha256": digest.hexdigest(),
            "train_prefix_length_from_filename": int(train_prefix),
            "first_anomaly_index_from_filename": int(first_anomaly),
            "source_rows": int(features.shape[0]),
            "source_n_var": int(features.shape[1]),
        }

        train = features[:train_prefix].copy()
        train_labels = np.zeros(train.shape[0], dtype=np.int64)
        test = features[train_prefix:].copy()
        test_labels = labels_numeric[train_prefix:].astype(np.int64, copy=False)
        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(
                train, train_labels
            )
        else:
            val, val_labels = test.copy(), test_labels.copy()
        return train, val, test, train_labels, val_labels, test_labels


class SWaTSegLoader(TSDataset):
    def __init__(self, cfg, split):
        # 新版本修复首行与攻击标签解析；使用独立缓存，禁止复用旧错位缓存。
        self.downsample_cache_tag = "swat_v2_aligned_any_label"
        super(SWaTSegLoader, self).__init__(cfg, split)

    @staticmethod
    def _read_source(csv_path, excel_path):
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        if os.path.exists(excel_path):
            frame = pd.read_excel(excel_path)
            frame.to_csv(csv_path, index=False)
            return frame
        raise FileNotFoundError(
            f"Neither {csv_path} nor {excel_path} exists."
        )

    @staticmethod
    def _canonical_label(value):
        return "".join(str(value).split()).lower()

    @classmethod
    def _prepare_normal_frame(cls, frame):
        frame = frame.copy()
        # Normal_v0/v1 的首行实际保存了字段名，而 Excel 自带列名是 F1/P1/...。
        first_row = [str(value).strip() for value in frame.iloc[0].tolist()]
        if (
            first_row
            and cls._canonical_label(first_row[0]) == "timestamp"
            and cls._canonical_label(first_row[-1]) == "normal/attack"
        ):
            frame.columns = first_row
            frame = frame.iloc[1:].reset_index(drop=True)
        frame.columns = [str(name).strip() for name in frame.columns]
        labels = frame.iloc[:, -1].map(cls._canonical_label)
        unknown = sorted(set(labels) - {"normal"})
        if unknown:
            raise ValueError(f"Unexpected SWaT normal labels: {unknown}")
        return frame

    @classmethod
    def _prepare_attack_frame(cls, frame):
        frame = frame.copy()
        # 部分 Attack_v0 CSV 将字段名作为第一条记录保存；识别后提升为正式表头，
        # 不能把 ``Normal/Attack`` 当成真实标签，也不能删除真正的首个样本。
        first_row = [str(value).strip() for value in frame.iloc[0].tolist()]
        if (
            first_row
            and cls._canonical_label(first_row[0]) == "timestamp"
            and cls._canonical_label(first_row[-1]) == "normal/attack"
        ):
            frame.columns = first_row
            frame = frame.iloc[1:].reset_index(drop=True)
        frame.columns = [str(name).strip() for name in frame.columns]
        labels = frame.iloc[:, -1].map(cls._canonical_label)
        unknown = sorted(set(labels) - {"normal", "attack"})
        if unknown:
            raise ValueError(f"Unexpected SWaT attack labels: {unknown}")
        return frame, (labels == "attack").to_numpy(dtype=np.int64)

    def _load_data(self):
        # ``SWaT_PGRF`` 仅改变论文复现实验协议，原始文件仍位于 SWaT 目录。
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, "SWaT")
        if self.downsample_rate > 1:
            try:
                return self._load_downsampled_data()
            except FileNotFoundError:
                print(f"downsampled data not found. Load original data instead and downsample.")

        train_csv_path = self.data_dir + "/SWaT_Dataset_Normal_v1.csv"
        train_excel_path = self.data_dir + "/SWaT_Dataset_Normal_v1.xlsx"

        train_source = self._read_source(train_csv_path, train_excel_path)
        train_source = self._prepare_normal_frame(train_source)
        train_df = train_source.iloc[:, 1:-1].astype(np.float32)

        test_csv_path = self.data_dir + "/SWaT_Dataset_Attack_v0.csv"
        test_excel_path = self.data_dir + "/SWaT_Dataset_Attack_v0.xlsx"

        test_source = self._read_source(test_csv_path, test_excel_path)
        test_source, test_labels = self._prepare_attack_frame(test_source)
        test_df = test_source.iloc[:, 1:-1].astype(np.float32)

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError(
                "SWaT train/test sensor columns or ordering do not match: "
                f"train={list(train_df.columns)}, test={list(test_df.columns)}"
            )

        self.var_names = list(train_df.columns)

        train = train_df.values
        train_labels = np.zeros(train.shape[0])
        test = test_df.values

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        if self.downsample_rate > 1:
            train, val, test = downsample(train, self.downsample_rate), downsample(val, self.downsample_rate), downsample(test, self.downsample_rate)
            train_labels = downsample_binary_labels(train_labels, self.downsample_rate)
            val_labels = downsample_binary_labels(val_labels, self.downsample_rate)
            test_labels = downsample_binary_labels(test_labels, self.downsample_rate)
            self._save_downsampled_data(train, val, test, train_labels, val_labels, test_labels)

        return train, val, test, train_labels, val_labels, test_labels


class WADISegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(WADISegLoader, self).__init__(cfg, split)

    def _load_data(self):
        if self.downsample_rate > 1:
            try:
                return self._load_downsampled_data()
            except:
                print(f"downsampled data not found. Load original data instead and downsample.")

        train_df = pd.read_csv(self.data_dir + "/WADI_14days_new.csv")
        train_df = train_df.dropna(axis='columns', how='all').dropna()
        train_df = train_df.iloc[:, 3:].astype(np.float32)
        test_df = pd.read_csv(self.data_dir + "/WADI_attackdataLABLE.csv", header=1)
        test_df = test_df.dropna(axis='columns', how='all').dropna()
        test_df = test_df.iloc[:, 3:].astype(np.float32)

        self.var_names = list(train_df.columns)

        test_labels = test_df['Attack LABLE (1:No Attack, -1:Attack)'].values
        test_labels = (test_labels == -1.0).astype(int)
        test = test_df.drop(columns='Attack LABLE (1:No Attack, -1:Attack)').values

        train = train_df.values
        train_labels = np.zeros(train.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        if self.downsample_rate > 1:
            train, val, test = downsample(train, self.downsample_rate), downsample(val, self.downsample_rate), downsample(test, self.downsample_rate)
            train_labels, val_labels, test_labels = downsample(train_labels, self.downsample_rate), downsample(val_labels, self.downsample_rate), downsample(test_labels, self.downsample_rate)
            self._save_downsampled_data(train, val, test, train_labels, val_labels, test_labels)

        return train, val, test, train_labels, val_labels, test_labels


class Lorenz96SegLoader(TSDataset):
    def __init__(self, cfg, split):
        self.anomaly_type, factor_str = cfg.DATA.NAME[len('Lorenz96_'):].rsplit('_', 1)
        self.factor = 'None' if factor_str == 'factorNone' else float(factor_str[len('factor'):])
        super(Lorenz96SegLoader, self).__init__(cfg, split)
        self.true_cm = np.load(os.path.join(self.data_dir, 'GC.npy'))

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'Lorenz96')
        train = np.load(os.path.join(self.data_dir, 'train.npy'))
        train_labels = np.zeros(train.shape[0])

        test = np.load(os.path.join(self.data_dir, f'test_{self.anomaly_type}_outliers_factor{self.factor}.npy'))

        test_labels = np.load(os.path.join(self.data_dir, f'test_{self.anomaly_type}_outliers_factor{self.factor}_labels.npy'))

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class VARSegLoader(TSDataset):
    def __init__(self, cfg, split):
        self.anomaly_type, factor_str = cfg.DATA.NAME[len('VAR_'):].rsplit('_', 1)
        self.factor = 'None' if factor_str == 'factorNone' else float(factor_str[len('factor'):])
        super(VARSegLoader, self).__init__(cfg, split)
        self.true_cm = np.load(os.path.join(self.data_dir, 'GC.npy'))

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'VAR')
        train = np.load(os.path.join(self.data_dir, 'train.npy'))
        train_labels = np.zeros(train.shape[0])

        test = np.load(os.path.join(self.data_dir, f'test_{self.anomaly_type}_outliers_factor{self.factor}.npy'))

        test_labels = np.load(os.path.join(self.data_dir, f'test_{self.anomaly_type}_outliers_factor{self.factor}_labels.npy'))

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


def build_dataset(cfg, split):
    dataset_name = cfg.DATA.NAME

    dataset_loaders = {
        "TSB_SWAT_1": TSBADMSegLoader,
        "SMD": SMDSegLoader,
        "MSL": MSLSegLoader,
        "SMAP": SMAPSegLoader,
        "PSM": PSMSegLoader,
        "SWaT": SWaTSegLoader,
        "WADI": WADISegLoader,
        "Lorenz96": Lorenz96SegLoader,
        "VAR": VARSegLoader,
    }

    for key in dataset_loaders:
        if key in dataset_name:
            return dataset_loaders[key](cfg, split)

    raise ValueError(f"Unknown dataset name: {dataset_name}")
