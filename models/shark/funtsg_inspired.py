"""Function/relationship edits inspired by Fun-TSG; not its official generator.

Only candidate construction changes. Normal TRAIN statistics, eligibility,
support, discrete-state semantics and acceptance remain the Graph-VAR teacher's.
These are local edits, not a structural causal model or a propagation simulator.
"""
import torch


@torch.no_grad()
def function_candidates(mechanism, window, seed, start, end, anomaly_type, strengths):
    strengths = torch.as_tensor(strengths, device=window.device, dtype=window.dtype).flatten()
    invalid = [None] * strengths.numel()
    if window.ndim != 2 or not 0 <= seed < window.size(1):
        return invalid
    if not 0 <= start < end <= window.size(0) or anomaly_type not in mechanism.SUPPORTED_TYPES:
        return invalid
    state = mechanism._state(window.device, window.dtype)
    if state is None or window.size(1) != mechanism.n_nodes or not state["valid_nodes_host"][seed]:
        return invalid
    if not torch.isfinite(strengths).all() or ((strengths < 0) | (strengths > 1)).any():
        raise ValueError("candidate strengths must be finite and in [0, 1]")
    # Global TRAIN mean/scale: never fit a mechanism on the edited short window.
    z = (window - state["mean"]) / state["scale"]
    offset = mechanism.lag_offset_steps
    first = max(start, mechanism.lags + (offset if anomaly_type == "lag_break" else 0))
    if first >= end:
        return invalid
    original = z[first:end, seed]
    if anomaly_type == "lag_break":
        # Shift the observed temporal function; no circular wrap or future reads.
        target = z[first - offset:end - offset, seed]
    elif anomaly_type == "neighbor_break":
        # Replace with another observed function in the seed's TRAIN units.
        # Prefer a non-parent donor of the same continuous/discrete kind.
        excluded = set(state["parent_index_host"][seed]) | {seed}
        donors = [(seed + step) % mechanism.n_nodes for step in range(1, mechanism.n_nodes)]
        donor = next((node for node in donors if node not in excluded
                      and state["valid_nodes_host"][node]
                      and state["discrete_mask_host"][node] == state["discrete_mask_host"][seed]), None)
        if donor is None:
            return invalid
        target = z[first:end, donor]
    elif anomaly_type == "relation_flip":
        positions = torch.arange(first, end, device=window.device)
        slot = mechanism._strongest_parent_slot(z, seed, first, end, state)
        if slot is None:
            return invalid
        parent = state["parent_index_host"][seed][slot]
        # Reflect the seed trajectory around its fitted parent contribution.
        reference = torch.zeros_like(original)
        for lag in range(1, mechanism.lags + 1):
            reference += z[positions - lag, parent] * state["parent_coef"][seed, slot, lag - 1]
        target = 2 * reference - original
    else:
        # Mutate local gain plus an additive time function, as in the old edits.
        direction = 1.0 if seed % 2 == 0 else -1.0
        ramp = torch.linspace(0, 1, end - first, device=window.device, dtype=window.dtype)
        target = (1 + .2 * direction) * original + direction * (.5 + ramp)
    delta = target - original
    values = window[start:end, seed].expand(strengths.numel(), -1).clone()
    local_start = first - start
    # Add to the original directly, so zero strength is an exact identity.
    values[:, local_start:] += strengths[:, None] * delta[None, :] * state["scale"][seed]
    results = []
    for strength, candidate in zip(strengths.tolist(), values):
        if strength > 0 and mechanism.adaptive and state["discrete_mask_host"][seed]:
            count = state["discrete_state_count_host"][seed]
            candidate = mechanism._discrete_counterfactual(window[start:end, seed], candidate,
                state["discrete_states"][seed, :count], local_start, end - start, strength)
        results.append(candidate.clone() if torch.isfinite(candidate).all() else None)
    return results
