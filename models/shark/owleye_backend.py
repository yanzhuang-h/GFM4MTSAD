import importlib.util
import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shark.gfm_backend import (
    GFMBackendContract,
    GFMBackendInput,
    GFMBackendSpec,
    NODE_ANOMALY_EVIDENCE_SEMANTICS,
    validate_backend_output,
)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_source_class(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"OWLEYE source model not found: {path}")
    module_name = f"_shark_owleye_source_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OWLEYE source model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "OWLEYE"):
        raise ImportError(f"OWLEYE class is missing from source model: {path}")
    return module.OWLEYE


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_source_model(checkpoint_path):
    # The canonical checkpoint is a symlink into runs/<run>/, so discovery
    # must walk upward from the resolved target rather than assuming that its
    # immediate parent is OWLEYE/checkpoints/.
    for parent in Path(checkpoint_path).resolve().parents:
        candidate = parent / "model.py"
        config = parent / "best_param_pt" / "OWLEYE.json"
        if candidate.exists() and config.exists():
            return candidate
    raise FileNotFoundError(
        "Could not discover OWLEYE/model.py from checkpoint path: "
        f"{checkpoint_path}"
    )


class OWLEYEBackend(nn.Module, GFMBackendContract):
    """Frozen OWLEYE dictionary runtime with SHARK's standard GFM outputs.

    The released checkpoint contains both the neural parameters and the four
    source-domain pattern dictionaries.  MTS sensor windows are interpreted as
    small graphs: the shared Token Bridge supplies 64D node attributes and the
    SHARK Pearson builder supplies graph structure.
    """

    def __init__(
        self,
        checkpoint_path,
        source_model_path=None,
        checkpoint_sha256=None,
        source_model_sha256=None,
        tune_input_prompt=True,
        tune_temperature=False,
        source_pattern_limit=64,
        in_context_nodes=10,
        in_context_selection="top_degree",
        truncation_ratio=0.1,
        eps=1e-12,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"OWLEYE checkpoint not found: {checkpoint_path}"
            )
        if checkpoint_sha256:
            actual_sha256 = _sha256(checkpoint_path)
            if actual_sha256 != str(checkpoint_sha256):
                raise ValueError(
                    "OWLEYE checkpoint SHA256 drifted: "
                    f"expected={checkpoint_sha256}, actual={actual_sha256}"
                )
        if source_model_path is None:
            source_model_path = _discover_source_model(checkpoint_path)
        source_model_path = Path(source_model_path).resolve()
        if source_model_sha256:
            actual_source_sha256 = _sha256(source_model_path)
            if actual_source_sha256 != str(source_model_sha256):
                raise ValueError(
                    "OWLEYE source model SHA256 drifted: "
                    f"expected={source_model_sha256}, "
                    f"actual={actual_source_sha256}"
                )

        payload = _load_checkpoint(checkpoint_path)
        required = {
            "model_state_dict",
            "model_config",
            "source_patterns",
            "source_struct_patterns",
            "train_datasets",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"OWLEYE checkpoint is missing fields: {missing}")

        model_config = dict(payload["model_config"])
        owleye_class = _load_source_class(source_model_path)
        self.owleye = owleye_class(device="cpu", **model_config)
        self.owleye.load_state_dict(payload["model_state_dict"], strict=True)

        self.token_dim = int(model_config["in_feats"])
        self.num_hops = int(model_config["num_hops"])
        self.embedding_dim = int(self.owleye.embedding_dim)
        self.struct_dim = int(self.owleye.st_dim * self.num_hops)
        self.beta = float(model_config.get("beta", 0.01))
        self.base_temperature = float(model_config["temperature"])
        self.tune_input_prompt = bool(tune_input_prompt)
        self.tune_temperature = bool(tune_temperature)
        self.source_pattern_limit = int(source_pattern_limit)
        self.in_context_nodes = int(in_context_nodes)
        self.in_context_selection = str(in_context_selection)
        self.truncation_ratio = float(truncation_ratio)
        self.eps = float(eps)
        if self.source_pattern_limit < 1:
            raise ValueError("OWLEYE source pattern limit must be positive")
        if self.in_context_nodes < 1:
            raise ValueError("OWLEYE in-context node count must be positive")
        if self.in_context_selection != "top_degree":
            raise ValueError(
                "OWLEYE currently supports deterministic top_degree "
                "in-context selection only"
            )
        if not 0.0 <= self.truncation_ratio < 1.0:
            raise ValueError(
                "OWLEYE truncation ratio must be in [0, 1)"
            )

        expected_source_sha256 = dict(
            payload.get("source_sha256", {})
        ).get("model.py")
        if expected_source_sha256:
            actual_source_sha256 = _sha256(source_model_path)
            if actual_source_sha256 != expected_source_sha256:
                raise ValueError(
                    "OWLEYE source model SHA256 drifted: "
                    f"expected={expected_source_sha256}, "
                    f"actual={actual_source_sha256}"
                )

        self.input_prompt = nn.Parameter(torch.zeros(self.token_dim))
        self.temperature_log_scale = nn.Parameter(torch.zeros(()))
        datasets = tuple(str(name) for name in payload["train_datasets"])
        if not datasets:
            raise ValueError("OWLEYE checkpoint contains no source domains")
        self.source_datasets = datasets
        self.source_pattern_original_counts = {}
        for index, dataset in enumerate(datasets):
            pattern = payload["source_patterns"][dataset].detach().float()
            struct_pattern = (
                payload["source_struct_patterns"][dataset].detach().float()
            )
            if pattern.dim() != 2 or pattern.size(1) != self.embedding_dim:
                raise ValueError(
                    f"Invalid OWLEYE pattern shape for {dataset}: "
                    f"{tuple(pattern.shape)}"
                )
            if (
                struct_pattern.dim() != 2
                or struct_pattern.size(1) != self.struct_dim
            ):
                raise ValueError(
                    f"Invalid OWLEYE structural pattern shape for {dataset}: "
                    f"{tuple(struct_pattern.shape)}"
                )
            if pattern.size(0) != struct_pattern.size(0):
                raise ValueError(
                    f"OWLEYE attribute/structure pattern counts differ for "
                    f"{dataset}: {pattern.size(0)} vs "
                    f"{struct_pattern.size(0)}"
                )
            self.source_pattern_original_counts[dataset] = int(
                pattern.size(0)
            )
            if pattern.size(0) > self.source_pattern_limit:
                selection = torch.linspace(
                    0,
                    pattern.size(0) - 1,
                    steps=self.source_pattern_limit,
                ).round().long()
                pattern = pattern.index_select(0, selection)
                struct_pattern = struct_pattern.index_select(0, selection)
            self.register_buffer(f"source_pattern_{index}", pattern)
            self.register_buffer(
                f"source_struct_pattern_{index}", struct_pattern
            )

        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.source_model_path = str(source_model_path)
        self.checkpoint_trial = int(payload.get("trial", -1))
        self.checkpoint_seed = int(payload.get("seed", -1))
        self.checkpoint_source_sha256 = dict(
            payload.get("source_sha256", {})
        )
        self.freeze_backbone()

    @classmethod
    def from_config(cls, cfg):
        settings = cfg.SHARK.GFM_BACKEND
        return cls(
            checkpoint_path=str(settings.CHECKPOINT_PATH),
            source_model_path=str(
                getattr(settings, "SOURCE_MODEL_PATH", "")
            ) or None,
            checkpoint_sha256=str(
                getattr(settings, "CHECKPOINT_SHA256", "")
            ) or None,
            source_model_sha256=str(
                getattr(settings, "SOURCE_MODEL_SHA256", "")
            ) or None,
            tune_input_prompt=bool(
                getattr(settings, "TRAIN_INPUT_PROMPT", True)
            ),
            tune_temperature=bool(
                getattr(settings, "TRAIN_TEMPERATURE", False)
            ),
            source_pattern_limit=int(
                getattr(settings, "SOURCE_PATTERN_LIMIT", 64)
            ),
            in_context_nodes=int(
                getattr(settings, "IN_CONTEXT_NODES", 10)
            ),
            in_context_selection=str(
                getattr(settings, "IN_CONTEXT_SELECTION", "top_degree")
            ),
            truncation_ratio=float(
                getattr(settings, "TRUNCATION_RATIO", 0.1)
            ),
        )

    def backend_spec(self):
        return GFMBackendSpec(
            name="owleye_frozen_dictionary_input_prompt_peft",
            token_dim=self.token_dim,
            embedding_dim=self.embedding_dim,
            uses_source_node_features=False,
            adaptation_strategy=(
                "degree_conditioned_zero_input_prompt"
                if self.tune_input_prompt
                else "temperature_log_scale"
                if self.tune_temperature
                else "frozen"
            ),
            uses_declared_node_tokens=True,
            node_score_semantics=NODE_ANOMALY_EVIDENCE_SEMANTICS,
        )

    def freeze_backbone(self):
        self.owleye.eval()
        for parameter in self.owleye.parameters():
            parameter.requires_grad = False
        self.input_prompt.requires_grad = self.tune_input_prompt
        self.temperature_log_scale.requires_grad = self.tune_temperature

    def adaptation_parameter_names(self):
        names = []
        if self.tune_input_prompt:
            names.append("input_prompt")
        if self.tune_temperature:
            names.append("temperature_log_scale")
        return tuple(names)

    def adaptation_is_identity(self):
        parameters = dict(self.named_parameters())
        return all(
            int(torch.count_nonzero(parameters[name].detach())) == 0
            for name in self.adaptation_parameter_names()
        )

    def effective_temperature(self):
        log_scale = self.temperature_log_scale.clamp(-4.0, 4.0)
        return self.base_temperature * torch.exp(log_scale)

    def prompt_diagnostics(self):
        return {
            "owleye_checkpoint": self.checkpoint_path,
            "owleye_checkpoint_sha256": self.checkpoint_sha256,
            "owleye_checkpoint_trial": self.checkpoint_trial,
            "owleye_checkpoint_seed": self.checkpoint_seed,
            "owleye_source_model": self.source_model_path,
            "owleye_source_model_sha256": _sha256(
                self.source_model_path
            ),
            "owleye_source_domains": list(self.source_datasets),
            "owleye_source_pattern_limit": self.source_pattern_limit,
            "owleye_source_pattern_original_counts": dict(
                self.source_pattern_original_counts
            ),
            "owleye_in_context_nodes": self.in_context_nodes,
            "owleye_in_context_selection": self.in_context_selection,
            "owleye_truncation_ratio": self.truncation_ratio,
            "owleye_base_temperature": self.base_temperature,
            "owleye_effective_temperature": float(
                self.effective_temperature().detach().cpu()
            ),
            "owleye_input_prompt_norm": float(
                self.input_prompt.detach().float().norm().cpu()
            ),
        }

    def _source_patterns(self):
        return [
            getattr(self, f"source_pattern_{index}")
            for index in range(len(self.source_datasets))
        ]

    def _source_struct_patterns(self):
        return [
            getattr(self, f"source_struct_pattern_{index}")
            for index in range(len(self.source_datasets))
        ]

    def _normalize_adjacency(self, adjacency):
        adjacency = adjacency.float().clamp_min(0.0)
        adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
        adjacency = adjacency.clone()
        diagonal = torch.arange(adjacency.size(1), device=adjacency.device)
        adjacency[:, diagonal, diagonal] = 0.0
        degree = adjacency.sum(dim=-1).clamp_min(self.eps)
        inv_sqrt = degree.rsqrt()
        return adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)

    def _apply_input_prompt(self, node_tokens, adjacency):
        graph = adjacency.float().clamp_min(0.0)
        graph = torch.maximum(graph, graph.transpose(1, 2))
        degree = graph.sum(dim=-1, keepdim=True)
        degree = degree / degree.amax(dim=1, keepdim=True).clamp_min(
            self.eps
        )
        return node_tokens + degree * self.input_prompt.view(1, 1, -1)

    def _encode_nodes(self, node_tokens, adjacency):
        propagated = [node_tokens.float()]
        for _ in range(self.num_hops):
            propagated.append(torch.bmm(adjacency, propagated[-1]))

        feature_states = propagated
        for index, layer in enumerate(self.owleye.layers):
            if index != 0:
                feature_states = [
                    self.owleye.dropout(value) for value in feature_states
                ]
            feature_states = [layer(value) for value in feature_states]
            if index != len(self.owleye.layers) - 1:
                feature_states = [
                    self.owleye.act(value) for value in feature_states
                ]
        first = feature_states[0]
        embeddings = torch.cat(
            [value - first for value in feature_states[1:]], dim=-1
        )

        struct = torch.ones(
            node_tokens.size(0),
            node_tokens.size(1),
            self.owleye.st_dim,
            dtype=node_tokens.dtype,
            device=node_tokens.device,
        )
        struct_states = []
        for index, layer in enumerate(self.owleye.struct_layers):
            struct = torch.bmm(adjacency, layer(struct))
            if index != len(self.owleye.struct_layers) - 1:
                struct = self.owleye.act(struct)
            struct_states.append(struct)
        first_struct = struct_states[0]
        struct_embeddings = torch.cat(
            [value - first_struct for value in struct_states[1:]],
            dim=-1,
        )
        return embeddings, struct_embeddings

    def _select_in_context(self, embeddings, struct_embeddings, adjacency):
        count = min(self.in_context_nodes, embeddings.size(1))
        degree = adjacency.sum(dim=-1)
        indices = torch.topk(degree, k=count, dim=-1).indices

        def gather(values):
            expanded = indices.unsqueeze(-1).expand(
                -1, -1, values.size(-1)
            )
            return torch.gather(values, dim=1, index=expanded)

        # The released evaluation path detaches target-graph patterns before
        # inserting them into its dictionary. Preserve that boundary here.
        return (
            gather(embeddings).detach(),
            gather(struct_embeddings).detach(),
            indices,
        )

    def _source_domain_similarity(self, support, query):
        transformed = self.owleye.domain_sim.struct_sim(support)
        logits = torch.einsum("kh,bnh->bkn", transformed, query)
        return F.softmax(logits, dim=1).amax(dim=1)

    def _batched_domain_similarity(self, support, query):
        transformed = self.owleye.domain_sim.struct_sim(support)
        logits = torch.bmm(transformed, query.transpose(1, 2))
        return F.softmax(logits, dim=1).amax(dim=1)

    def _truncate_attention(self, scores):
        count = int(scores.size(-1) * self.truncation_ratio)
        if count > 0:
            indices = torch.topk(
                scores, count, dim=-1, largest=False
            ).indices
            scores = scores.scatter(
                dim=-1,
                index=indices,
                value=float("-inf"),
            )
        return scores

    def _source_attention(self, query, support, query_layer, key_layer):
        q = F.leaky_relu(query_layer(query))
        k = F.leaky_relu(key_layer(support))
        scores = torch.einsum("bnh,kh->bnk", q, k)
        scores = scores / math.sqrt(self.embedding_dim)
        scores = scores / self.effective_temperature()
        scores = self._truncate_attention(scores)
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bnk,kh->bnh", weights, support)

    def _batched_attention(self, query, support, query_layer, key_layer):
        q = F.leaky_relu(query_layer(query))
        k = F.leaky_relu(key_layer(support))
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / math.sqrt(self.embedding_dim)
        scores = scores / self.effective_temperature()
        scores = self._truncate_attention(scores)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, support)

    def _reconstruct(
        self,
        query,
        struct_query,
        target_patterns,
        target_struct_patterns,
    ):
        source_patterns = self._source_patterns()
        source_struct_patterns = self._source_struct_patterns()
        domain_similarities = [
            self._source_domain_similarity(support, struct_query)
            for support in source_struct_patterns
        ]
        domain_similarities.append(
            self._batched_domain_similarity(
                target_struct_patterns, struct_query
            )
        )
        domain_similarity = torch.stack(domain_similarities, dim=-1)

        feature_reconstruction = torch.zeros_like(query)
        struct_reconstruction = torch.zeros_like(struct_query)
        for index, (support, struct_support) in enumerate(
            zip(source_patterns, source_struct_patterns)
        ):
            weight = domain_similarity[..., index].unsqueeze(-1)
            feature_reconstruction = feature_reconstruction + weight * (
                self._source_attention(
                    query, support, self.owleye.Wq, self.owleye.Wk
                )
            )
            struct_reconstruction = struct_reconstruction + weight * (
                self._source_attention(
                    struct_query,
                    struct_support,
                    self.owleye.Wq_struct,
                    self.owleye.Wk_struct,
                )
            )

        target_index = len(source_patterns)
        target_weight = domain_similarity[..., target_index].unsqueeze(-1)
        feature_reconstruction = feature_reconstruction + target_weight * (
            self._batched_attention(
                query,
                target_patterns,
                self.owleye.Wq,
                self.owleye.Wk,
            )
        )
        struct_reconstruction = struct_reconstruction + target_weight * (
            self._batched_attention(
                struct_query,
                target_struct_patterns,
                self.owleye.Wq_struct,
                self.owleye.Wk_struct,
            )
        )
        denominator = float(len(source_patterns) + 1)
        return (
            feature_reconstruction / denominator,
            struct_reconstruction / denominator,
            domain_similarity,
        )

    def forward_backend(self, inputs: GFMBackendInput):
        adjacency = self._normalize_adjacency(inputs.adjacency)
        node_tokens = self._apply_input_prompt(
            inputs.node_tokens, inputs.adjacency
        )
        embeddings, struct_embeddings = self._encode_nodes(
            node_tokens, adjacency
        )
        target_patterns, target_struct_patterns, target_indices = (
            self._select_in_context(
                embeddings, struct_embeddings, adjacency
            )
        )
        reconstructed, reconstructed_struct, domain_similarity = (
            self._reconstruct(
                embeddings,
                struct_embeddings,
                target_patterns,
                target_struct_patterns,
            )
        )
        node_scores = torch.linalg.vector_norm(
            reconstructed - embeddings, dim=-1
        ) + self.beta * torch.linalg.vector_norm(
            reconstructed_struct - struct_embeddings, dim=-1
        )
        outputs = {
            "node_scores": node_scores.clamp_min(0.0),
            "node_embeddings": embeddings,
            "graph_embedding": embeddings.mean(dim=1),
            "owleye_reconstructed_embeddings": reconstructed,
            "owleye_struct_embeddings": struct_embeddings,
            "owleye_domain_similarity": domain_similarity,
            "owleye_in_context_indices": target_indices,
            "owleye_effective_temperature": self.effective_temperature(),
        }
        return validate_backend_output(outputs, inputs, self.backend_spec())

    def train(self, mode=True):
        super().train(mode)
        self.owleye.eval()
        return self
