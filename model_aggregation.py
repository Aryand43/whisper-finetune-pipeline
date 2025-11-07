import torch
import os
from typing import List, Optional, OrderedDict


# -----------------------------
# Checkpoint Averaging
# -----------------------------
def average_checkpoints(
    state_dicts: List[str],
    weights: Optional[List[float]] = None,
    use_float64: bool = True,
) -> OrderedDict:
    """Exact key match, CPU only, optional float64 accumulation."""
    assert len(state_dicts) >= 1, "need at least one state_dict"
    ref = state_dicts[0]
    ref_keys = list(ref.keys())

    # key and shape match
    for i, sd in enumerate(state_dicts[1:], 1):
        assert set(sd.keys()) == set(ref_keys), f"key mismatch at index {i}"
        for k in ref_keys:
            assert sd[k].shape == ref[k].shape, f"shape mismatch for {k} at index {i}"

    # normalize weights
    assert abs(float(sum(weights))) - 1 < 1e-6, "weights must sum to 1"

    # initialize accumulator
    out = OrderedDict()
    acc = {}
    # Fill up accumulator with zeros
    for k, t0 in ref.items():
        if t0.is_floating_point():
            dtype = torch.float64 if use_float64 else torch.float32
            acc[k] = torch.zeros_like(t0.cpu(), dtype=dtype)
        else:
            acc[k] = None

    # accumulate
    for sd, w in zip(state_dicts, weights):
        for k, t in sd.items():
            if t.is_floating_point():
                acc[k].add_(t.cpu().to(dtype=acc[k].dtype), alpha=w)

    # finalize output by casting back to original dtypes
    for k, t0 in ref.items():
        if t0.is_floating_point():
            out[k] = acc[k].to(dtype=t0.dtype)
        else:
            out[k] = t0.clone().cpu()

    return out
