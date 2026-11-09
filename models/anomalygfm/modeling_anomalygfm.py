from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency, eps=1e-6):
    degree = adjacency.sum(dim=-1).clamp_min(eps)
    degree_inv_sqrt = degree.pow(-0.5)
    return adjacency * degree_inv_sqrt.unsqueeze(-1) * degree_inv_sqrt.unsqueeze(-2)


class LowRankNodeAdapter(nn.Module):
    """Residual PEFT adapter for GFM node embeddings."""

    def __init__(self, feature_dim, rank, scale, dropout):
        super().__init__()
        self.scale = float(scale)
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        delta = self.up(F.gelu(self.down(self.dropout(features))))
        return features + self.scale * delta


class AnomalyGFM(nn.Module):
    def __init__(self, cfg, initialization="pretrained"):
        super(AnomalyGFM, self).__init__()
        if initialization not in ("pretrained", "random"):
            raise ValueError(
                "AnomalyGFM initialization must be 'pretrained' or 'random', "
                f"got: {initialization}"
            )
        self.cfg = cfg
        self.initialization = initialization
        self.beta = cfg.ANOMALYGFM.BETA
        self.embedding_dim = cfg.ANOMALYGFM.EMBEDDING_DIM
        self.input_dim = cfg.ANOMALYGFM.INPUT_DIM
        self.model = self._build_model()
        self.checkpoint_report = self._initialize_model()
        self.normal_prompt_raw = nn.Parameter(torch.randn(self.embedding_dim), requires_grad=False)
        self.abnormal_prompt_raw = nn.Parameter(torch.randn(self.embedding_dim), requires_grad=False)
        tuning_cfg = cfg.SHARK.GFM_TUNING
        if tuning_cfg.STRATEGY == "sensor_graph_prompt_adapter":
            self.node_adapter = LowRankNodeAdapter(
                self.embedding_dim,
                int(tuning_cfg.NODE_ADAPTER_RANK),
                float(tuning_cfg.NODE_ADAPTER_SCALE),
                float(tuning_cfg.NODE_ADAPTER_DROPOUT),
            )
        else:
            self.node_adapter = nn.Identity()
        self.register_buffer("normal_prompt_initial", self.normal_prompt_raw.detach().clone())
        self.register_buffer("abnormal_prompt_initial", self.abnormal_prompt_raw.detach().clone())
        self.freeze()
        self.configure_tuning()

    def _build_model(self):
        from models.anomalygfm.source import load_model_class
        Model = load_model_class(self.cfg.ANOMALYGFM.SOURCE_MODEL_PATH)

        return Model(
            self.input_dim,
            self.input_dim,
            self.embedding_dim,
            "prelu",
            self.cfg.ANOMALYGFM.NEGSAMP_RATIO,
            self.cfg.ANOMALYGFM.READOUT,
        )

    def _checkpoint_path(self):
        path = Path(self.cfg.ANOMALYGFM.CHECKPOINT_PATH)
        if path.is_absolute():
            return path
        shark_root = Path(__file__).resolve().parents[2]
        return shark_root / path

    def _initialize_model(self):
        if self.initialization == "pretrained":
            report = self._load_checkpoint()
        else:
            path = self._checkpoint_path()
            report = {
                "path": str(path),
                "found": path.exists(),
                "loaded": 0,
                "skipped": 0,
            }
        report["initialization"] = self.initialization
        return report

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        report = {
            "path": str(path),
            "found": path.exists(),
            "loaded": 0,
            "skipped": 0,
        }
        if not path.exists():
            raise FileNotFoundError(f"AnomalyGFM checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        current = self.model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                compatible[key] = value
            elif key in current:
                skipped.append(key)

        if skipped and not self.cfg.ANOMALYGFM.ALLOW_PARTIAL_LOAD:
            details = ", ".join(skipped[:5])
            raise RuntimeError(f"AnomalyGFM checkpoint has incompatible tensors: {details}")

        self.model.load_state_dict(compatible, strict=False)
        report["loaded"] = len(compatible)
        report["skipped"] = len(skipped)
        return report

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def configure_tuning(self):
        tuning_cfg = getattr(self.cfg.SHARK, "GFM_TUNING", None)
        if tuning_cfg is None or not tuning_cfg.ENABLE:
            return

        strategy = tuning_cfg.STRATEGY
        valid_strategies = {
            "frozen",
            "prompt",
            "prompt_only",
            "prompt_projection",
            "last_layer",
            "prompt_last_layer",
            "sensor_graph_prompt",
            "sensor_graph_prompt_anomaly",
            "sensor_graph_prompt_prompt",
            "sensor_graph_prompt_adapter",
        }
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported SHARK.GFM_TUNING.STRATEGY: {strategy}")

        train_prompts = bool(tuning_cfg.TRAIN_PROMPTS)
        train_prompt_projections = bool(tuning_cfg.TRAIN_PROMPT_PROJECTIONS)
        train_last_layer = bool(tuning_cfg.TRAIN_LAST_LAYER)
        train_node_adapter = False
        if strategy == "frozen":
            train_prompts = False
            train_prompt_projections = False
            train_last_layer = False
        elif strategy == "prompt_only":
            train_prompts = True
            train_prompt_projections = False
            train_last_layer = False
        elif strategy == "prompt_projection":
            train_prompts = True
            train_prompt_projections = True
            train_last_layer = False
        elif strategy == "last_layer":
            train_prompts = False
            train_prompt_projections = False
            train_last_layer = True
        elif strategy == "prompt_last_layer":
            train_prompts = True
            train_prompt_projections = True
            train_last_layer = True
        elif strategy == "sensor_graph_prompt":
            train_prompts = False
            train_prompt_projections = False
            train_last_layer = False
        elif strategy in ("sensor_graph_prompt_anomaly", "sensor_graph_prompt_prompt"):
            train_prompts = True
            train_prompt_projections = False
            train_last_layer = False
        elif strategy == "sensor_graph_prompt_adapter":
            train_prompts = True
            train_prompt_projections = False
            train_last_layer = False
            train_node_adapter = True

        if train_prompts:
            self.normal_prompt_raw.requires_grad = True
            self.abnormal_prompt_raw.requires_grad = True

        if train_prompt_projections:
            self._set_requires_grad_by_keywords(
                ("fc_normal_prompt", "fc_abnormal_prompt"),
                True,
            )

        if train_last_layer:
            keywords = tuple(tuning_cfg.UNFREEZE_KEYWORDS)
            if not keywords:
                keywords = ("gcn2", "fc1", "fc2")
            self._set_requires_grad_by_keywords(keywords, True)

        if train_node_adapter:
            for param in self.node_adapter.parameters():
                param.requires_grad = True

    def _set_requires_grad_by_keywords(self, keywords, requires_grad):
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in keywords):
                param.requires_grad = requires_grad

    def trainable_parameter_names(self):
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        return names

    def prompt_diagnostics(self):
        with torch.no_grad():
            normal = self.normal_prompt_raw.detach()
            abnormal = self.abnormal_prompt_raw.detach()
            normal_initial = self.normal_prompt_initial.to(device=normal.device, dtype=normal.dtype)
            abnormal_initial = self.abnormal_prompt_initial.to(device=abnormal.device, dtype=abnormal.dtype)
            normal_drift = (normal - normal_initial).norm()
            abnormal_drift = (abnormal - abnormal_initial).norm()
            prompt_cosine = F.cosine_similarity(normal.view(1, -1), abnormal.view(1, -1), dim=1)[0]
        diagnostics = {
            "gfm_initialization": self.initialization,
            "gfm_checkpoint_tensors_loaded": int(
                self.checkpoint_report.get("loaded", 0)
            ),
            "gfm_normal_prompt_drift": float(normal_drift.cpu().item()),
            "gfm_abnormal_prompt_drift": float(abnormal_drift.cpu().item()),
            "gfm_prompt_cosine": float(prompt_cosine.cpu().item()),
            "gfm_node_adapter_enabled": isinstance(
                self.node_adapter, LowRankNodeAdapter
            ),
        }
        if isinstance(self.node_adapter, LowRankNodeAdapter):
            diagnostics["gfm_node_adapter_down_norm"] = float(
                self.node_adapter.down.weight.detach().norm().cpu().item()
            )
            diagnostics["gfm_node_adapter_up_norm"] = float(
                self.node_adapter.up.weight.detach().norm().cpu().item()
            )
        return diagnostics

    def forward(self, node_features, adjacency):
        adjacency_norm = normalize_adjacency(adjacency)
        outputs = []
        for idx in range(node_features.size(0)):
            outputs.append(self._forward_single(node_features[idx:idx + 1], adjacency_norm[idx:idx + 1], adjacency[idx]))
        node_scores = torch.cat([output["node_scores"] for output in outputs], dim=0)
        graph_embedding = torch.cat([output["graph_embedding"] for output in outputs], dim=0)
        node_embeddings = torch.cat([output["node_embeddings"] for output in outputs], dim=0)
        return {
            "node_scores": node_scores,
            "graph_embedding": graph_embedding,
            "node_embeddings": node_embeddings,
        }

    def _forward_single(self, node_features, adjacency_norm, raw_adjacency):
        _, _, _, emb_residual, normal_prompt, abnormal_prompt, _ = self.model(
            node_features,
            adjacency_norm,
            raw_adjacency,
            self.normal_prompt_raw,
            self.abnormal_prompt_raw,
        )
        emb_residual = self._as_node_embeddings(emb_residual, node_features)
        emb_residual = self.node_adapter(emb_residual)
        emb_residual = F.normalize(emb_residual, p=2, dim=-1)
        normal_prompt = F.normalize(normal_prompt, p=2, dim=0)
        abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
        score_normal = torch.matmul(emb_residual, normal_prompt)
        score_abnormal = torch.matmul(emb_residual, abnormal_prompt)
        node_scores = torch.exp(score_abnormal) + self.beta * torch.exp(-score_normal)
        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(0)
        return {
            "node_scores": node_scores,
            "node_embeddings": emb_residual,
            "graph_embedding": emb_residual.mean(dim=1),
        }

    @staticmethod
    def _as_node_embeddings(embeddings, node_features):
        batch_size = node_features.size(0)
        n_nodes = node_features.size(1)
        expected_hidden = embeddings.size(-1) if embeddings.dim() > 0 else "unknown"

        if embeddings.dim() == 3:
            if embeddings.size(0) == batch_size and embeddings.size(1) == n_nodes:
                return embeddings
            if batch_size == 1 and embeddings.size(0) == 1 and embeddings.size(1) == n_nodes:
                return embeddings

        if embeddings.dim() == 2:
            if embeddings.size(0) == n_nodes:
                return embeddings.unsqueeze(0)

        raise ValueError(
            "AnomalyGFM expected residual node embeddings with true node dimension. "
            f"Allowed shapes are [B, N, H], [1, N, H] for a single-window call, or [N, H]. "
            f"Got embeddings={tuple(embeddings.shape)}, node_features={tuple(node_features.shape)}, "
            f"expected_batch={batch_size}, expected_nodes={n_nodes}, embedding_dim={expected_hidden}. "
            "Refusing to expand graph-level/shared embeddings into per-node embeddings."
        )
