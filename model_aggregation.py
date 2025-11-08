from collections import OrderedDict
from typing import List, Optional

import torch


# -----------------------------
# Checkpoint Averaging
# -----------------------------
def average_checkpoints(
    checkpoint_paths: List[str],
    weights: Optional[List[float]] = None,
    use_float64: bool = True,
) -> OrderedDict:
    """Average OpenAI Whisper checkpoints without loading them all at once."""

    assert checkpoint_paths, "need at least one checkpoint"

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    assert len(weights) == len(checkpoint_paths), "weights must match checkpoints"

    weight_total = float(sum(weights))
    assert abs(weight_total - 1.0) < 1e-6, "weights must sum to 1"

    def _load_state_dict(path: str) -> OrderedDict:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        ordered = OrderedDict()
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                ordered[key] = value.cpu()
            else:
                ordered[key] = value
        return ordered

    ref_state = _load_state_dict(checkpoint_paths[0])
    ref_keys = list(ref_state.keys())

    acc = {}
    for key, tensor in ref_state.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            dtype = torch.float64 if use_float64 else torch.float32
            acc[key] = torch.zeros_like(tensor, dtype=dtype)
        else:
            acc[key] = None

    for idx, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
        state_dict = ref_state if idx == 0 else _load_state_dict(checkpoint_path)

        if idx > 0:
            assert set(state_dict.keys()) == set(ref_keys), f"key mismatch at index {idx}"
            for key in ref_keys:
                assert state_dict[key].shape == ref_state[key].shape, (
                    f"shape mismatch for {key} at index {idx}"
                )

        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                acc[key].add_(tensor.to(acc[key].dtype), alpha=weight)

        if idx > 0:
            del state_dict

    out = OrderedDict()
    for key, tensor in ref_state.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            out[key] = acc[key].to(dtype=tensor.dtype)
        else:
            out[key] = tensor.clone() if isinstance(tensor, torch.Tensor) else tensor

    return out
