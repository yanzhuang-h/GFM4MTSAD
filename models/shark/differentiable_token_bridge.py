import torch
import torch.nn as nn


class DifferentiableTokenBridge(nn.Module):
    """Permutation-equivariant linear bridge with graph-wise normalization."""

    def __init__(self, source_dim, token_dim, eps=1e-5):
        super().__init__()
        self.source_dim = int(source_dim)
        self.token_dim = int(token_dim)
        self.eps = float(eps)
        if self.source_dim < self.token_dim:
            raise ValueError("Token bridge source_dim must be >= token_dim")
        self.proj = nn.Linear(self.source_dim, self.token_dim, bias=False)
        self._init_slice_projection()

    def _init_slice_projection(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, : self.token_dim] = torch.eye(self.token_dim)

    def forward(self, source_node_features):
        if source_node_features.dim() != 3:
            raise ValueError("Token bridge input must have shape [B, N, D]")
        if source_node_features.size(-1) != self.source_dim:
            raise ValueError(
                f"Token bridge expected {self.source_dim}D source features, "
                f"got {source_node_features.size(-1)}D"
            )
        if not torch.isfinite(source_node_features).all():
            raise ValueError("Token bridge source features must be finite")
        tokens = self.proj(source_node_features.float())
        mean = tokens.mean(dim=-2, keepdim=True)
        variance = (tokens - mean).square().mean(dim=-2, keepdim=True)
        return (tokens - mean) / torch.sqrt(variance + self.eps)

    @torch.no_grad()
    def _pca_components_from_moments(
        self,
        count,
        feature_sum,
        feature_cross_product,
    ):
        count = int(count)
        if count < 2:
            raise ValueError("Token bridge PCA initialization needs at least two rows")
        feature_sum = feature_sum.detach().double().cpu()
        feature_cross_product = feature_cross_product.detach().double().cpu()
        if tuple(feature_sum.shape) != (self.source_dim,):
            raise ValueError("Token bridge feature_sum shape is invalid")
        if tuple(feature_cross_product.shape) != (
            self.source_dim,
            self.source_dim,
        ):
            raise ValueError("Token bridge cross-product shape is invalid")
        covariance = (
            feature_cross_product
            - torch.outer(feature_sum, feature_sum) / float(count)
        ) / float(count - 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[: self.token_dim]
        components = eigenvectors[:, order].transpose(0, 1).contiguous()

        pivot_index = components.abs().argmax(dim=1, keepdim=True)
        pivots = torch.gather(components, 1, pivot_index).squeeze(1)
        signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
        return components * signs[:, None]

    @torch.no_grad()
    def fit_pca_from_moments(self, count, feature_sum, feature_cross_product):
        components = self._pca_components_from_moments(
            count,
            feature_sum,
            feature_cross_product,
        )
        self.proj.weight.copy_(
            components.to(
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
        )

    @torch.no_grad()
    def refit_aligned_pca_from_moments(
        self,
        count,
        feature_sum,
        feature_cross_product,
    ):
        """Track a moving source subspace without returning to epoch-0 axes.

        The new PCA row space follows the current train-normal frontend.  Its
        orthonormal basis is then rotated to be the closest basis to the
        bridge immediately before the refit.  This is a moving-subspace
        update, not a static token-coordinate target.
        """
        components = self._pca_components_from_moments(
            count,
            feature_sum,
            feature_cross_product,
        )
        previous = self.proj.weight.detach().double().cpu()
        previous_basis = torch.linalg.qr(
            previous.transpose(0, 1),
            mode="reduced",
        ).Q.transpose(0, 1)
        cross = previous_basis @ components.transpose(0, 1)
        left, singular_values, right_transpose = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        rotation = left @ right_transpose
        aligned = rotation @ components
        previous_normalized = previous / previous.norm(dim=1, keepdim=True).clamp_min(
            self.eps
        )
        aligned_normalized = aligned / aligned.norm(dim=1, keepdim=True).clamp_min(
            self.eps
        )
        row_cosine = (previous_normalized * aligned_normalized).sum(dim=1)
        self.proj.weight.copy_(
            aligned.to(
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
        )
        return {
            "row_cosine_mean": float(row_cosine.mean()),
            "row_cosine_min": float(row_cosine.min()),
            "subspace_overlap_mean": float(singular_values.mean()),
            "subspace_overlap_min": float(singular_values.min()),
            "orthogonality_error": float(
                (
                    aligned @ aligned.transpose(0, 1)
                    - torch.eye(self.token_dim, dtype=aligned.dtype)
                ).abs().max()
            ),
        }

    @torch.no_grad()
    def refit_polar_aligned_pca_from_moments(
        self,
        count,
        feature_sum,
        feature_cross_product,
    ):
        """Track the current PCA row space in the nearest Bridge orientation.

        Unlike QR, the rectangular polar factor ``U @ Vh`` is the unique
        nearest row-orthonormal matrix when the Bridge has full row rank.  It
        therefore preserves the orientation carried by the trainable Bridge
        before aligning the new PCA basis.
        """
        components = self._pca_components_from_moments(
            count,
            feature_sum,
            feature_cross_product,
        )
        previous = self.proj.weight.detach().double().cpu()
        previous_left, previous_singular, previous_right_transpose = (
            torch.linalg.svd(previous, full_matrices=False)
        )
        if float(previous_singular.min()) <= self.eps:
            raise RuntimeError("Token bridge is rank deficient before polar refit")
        previous_polar = previous_left @ previous_right_transpose
        cross = previous_polar @ components.transpose(0, 1)
        left, subspace_singular, right_transpose = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        rotation = left @ right_transpose
        aligned = rotation @ components
        previous_normalized = previous / previous.norm(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)
        row_cosine = (previous_normalized * aligned).sum(dim=1)
        update_frobenius = (aligned - previous_polar).square().sum().sqrt()
        self.proj.weight.copy_(
            aligned.to(
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
        )
        return {
            "row_cosine_mean": float(row_cosine.mean()),
            "row_cosine_min": float(row_cosine.min()),
            "subspace_overlap_mean": float(subspace_singular.mean()),
            "subspace_overlap_min": float(subspace_singular.min()),
            "previous_rank_min_singular": float(previous_singular.min()),
            "polar_update_frobenius": float(update_frobenius),
            "orthogonality_error": float(
                (
                    aligned @ aligned.transpose(0, 1)
                    - torch.eye(self.token_dim, dtype=aligned.dtype)
                ).abs().max()
            ),
        }

    @torch.no_grad()
    def refit_trust_region_polar_pca_from_moments(
        self,
        count,
        feature_sum,
        feature_cross_product,
        max_rms_step=0.25,
    ):
        """Follow the polar-aligned PCA target with a bounded Stiefel step."""
        max_rms_step = float(max_rms_step)
        if not 0.0 < max_rms_step <= 1.0:
            raise ValueError("max_rms_step must be in (0, 1]")
        components = self._pca_components_from_moments(
            count,
            feature_sum,
            feature_cross_product,
        )
        previous = self.proj.weight.detach().double().cpu()
        previous_left, previous_singular, previous_right_transpose = (
            torch.linalg.svd(previous, full_matrices=False)
        )
        if float(previous_singular.min()) <= self.eps:
            raise RuntimeError("Token bridge is rank deficient before trust-region refit")
        previous_polar = previous_left @ previous_right_transpose
        cross = previous_polar @ components.transpose(0, 1)
        left, subspace_singular, right_transpose = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        target = (left @ right_transpose) @ components
        normalizer = float(self.token_dim) ** 0.5

        def retract(weight):
            retract_left, retract_singular, retract_right = torch.linalg.svd(
                weight,
                full_matrices=False,
            )
            if float(retract_singular.min()) <= self.eps:
                raise RuntimeError("trust-region interpolation became rank deficient")
            return retract_left @ retract_right

        def rms_step(candidate):
            return float(
                (candidate - previous_polar).square().sum().sqrt() / normalizer
            )

        target_rms_step = rms_step(target)
        if target_rms_step <= max_rms_step:
            interpolation_weight = 1.0
            updated = target
        else:
            lower = 0.0
            upper = 1.0
            updated = previous_polar
            for _ in range(32):
                middle = 0.5 * (lower + upper)
                candidate = retract(
                    (1.0 - middle) * previous_polar + middle * target
                )
                if rms_step(candidate) <= max_rms_step:
                    lower = middle
                    updated = candidate
                else:
                    upper = middle
            interpolation_weight = lower

        applied_rms_step = rms_step(updated)
        if applied_rms_step > max_rms_step + 1e-8:
            raise RuntimeError("trust-region Bridge step exceeded its registered radius")
        previous_normalized = previous / previous.norm(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)
        row_cosine = (previous_normalized * updated).sum(dim=1)
        target_row_cosine = (previous_normalized * target).sum(dim=1)
        self.proj.weight.copy_(
            updated.to(
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
        )
        return {
            "row_cosine_mean": float(row_cosine.mean()),
            "row_cosine_min": float(row_cosine.min()),
            "target_row_cosine_mean": float(target_row_cosine.mean()),
            "target_row_cosine_min": float(target_row_cosine.min()),
            "subspace_overlap_mean": float(subspace_singular.mean()),
            "subspace_overlap_min": float(subspace_singular.min()),
            "previous_rank_min_singular": float(previous_singular.min()),
            "target_polar_update_frobenius": float(
                (target - previous_polar).square().sum().sqrt()
            ),
            "polar_update_frobenius": float(
                (updated - previous_polar).square().sum().sqrt()
            ),
            "normalized_target_step": target_rms_step,
            "normalized_applied_step": applied_rms_step,
            "trust_region_radius": max_rms_step,
            "interpolation_weight": float(interpolation_weight),
            "orthogonality_error": float(
                (
                    updated @ updated.transpose(0, 1)
                    - torch.eye(self.token_dim, dtype=updated.dtype)
                ).abs().max()
            ),
        }


def build_differentiable_token_bridge(cfg, source_dim, token_dim):
    settings = cfg.SHARK.TOKEN_BRIDGE
    if str(settings.NAME) != "DifferentiableTokenBridge":
        raise ValueError(f"Unsupported token bridge: {settings.NAME}")
    return DifferentiableTokenBridge(
        source_dim=source_dim,
        token_dim=token_dim,
        eps=float(settings.EPS),
    )
