import torch
import os

# -----------------------------
# Flatten Nested State Dicts
# -----------------------------
def flatten_state_dict(state, parent_key='', sep='.'):
    items = {}
    for k, v in state.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_state_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# -----------------------------
# Checkpoint Averaging
# -----------------------------
def average_checkpoints(model_paths, weights=None, save_path="aggregated_model.bin"):
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    assert len(weights) == len(model_paths), "Weights must match number of models"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    avg_state_dict = None

    for model_path, weight in zip(model_paths, weights):
        loaded = torch.load(model_path, map_location="cpu")

        if "state_dict" in loaded:
            if "model" in loaded["state_dict"]:
                state_dict = loaded["state_dict"]["model"]
            else:
                state_dict = loaded["state_dict"]
        else:
            state_dict = loaded

        state_dict = flatten_state_dict(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() * weight for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                if k in state_dict:
                    avg_state_dict[k] += state_dict[k].float() * weight

    torch.save(avg_state_dict, save_path)
    print(f"Averaged model saved at: {save_path}")
