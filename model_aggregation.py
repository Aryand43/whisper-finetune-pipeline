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

# -----------------------------
# Average nested state dicts while preserving structure
# -----------------------------
def average_nested_state_dict(state_dicts, weights):
    """Average state dicts while preserving nested structure"""
    if not state_dicts:
        return {}
    
    avg_state_dict = {}
    
    def recursive_average(dicts, weights, result_dict):
        for key in dicts[0].keys():
            if isinstance(dicts[0][key], dict):
                result_dict[key] = {}
                recursive_average([d[key] for d in dicts], weights, result_dict[key])
            elif isinstance(dicts[0][key], torch.Tensor):
                # Average the tensors
                weighted_sum = sum(d[key].float() * w for d, w in zip(dicts, weights))
                result_dict[key] = weighted_sum
            else:
                # For non-tensor, non-dict values, take from first dict
                result_dict[key] = dicts[0][key]
    
    recursive_average(state_dicts, weights, avg_state_dict)
    return avg_state_dict

def average_checkpoints_structured(model_paths, weights=None, save_path="aggregated_model_structured.bin"):
    """Structure-preserving version of checkpoint averaging"""
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    assert len(weights) == len(model_paths), "Weights must match number of models"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    state_dicts = []
    
    for model_path in model_paths:
        loaded = torch.load(model_path, map_location="cpu")

        if "state_dict" in loaded:
            if "model" in loaded["state_dict"]:
                state_dict = loaded["state_dict"]["model"]
            else:
                state_dict = loaded["state_dict"]
        else:
            state_dict = loaded
        
        state_dicts.append(state_dict)
    
    # Average while preserving structure
    avg_state_dict = average_nested_state_dict(state_dicts, weights)
    
    torch.save(avg_state_dict, save_path)
    print(f"Structure-preserving averaged model saved at: {save_path}")
    return avg_state_dict


if __name__ == "__main__":
    print("Testing structure-preserving vs flattening approach...")
    
    # Create two mock nested state dictionaries
    state_dict1 = {
        "encoder": {
            "layer1": {
                "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "bias": torch.tensor([0.5, 1.5])
            },
            "layer2": {
                "weight": torch.tensor([[5.0, 6.0]]),
                "bias": torch.tensor([2.5])
            }
        },
        "decoder": {
            "output": {
                "weight": torch.tensor([[7.0, 8.0], [9.0, 10.0]])
            }
        }
    }
    
    state_dict2 = {
        "encoder": {
            "layer1": {
                "weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
                "bias": torch.tensor([1.0, 2.0])
            },
            "layer2": {
                "weight": torch.tensor([[6.0, 7.0]]),
                "bias": torch.tensor([3.0])
            }
        },
        "decoder": {
            "output": {
                "weight": torch.tensor([[8.0, 9.0], [10.0, 11.0]])
            }
        }
    }
    
    print("\n=== Original State Dict 1 ===")
    print(state_dict1)
    
    print("\n=== Original State Dict 2 ===")
    print(state_dict2)

    
    # Test structure-preserving averaging
    print("\n=== Structure-Preserving Average (weights: [0.6, 0.4]) ===")
    averaged_structured = average_nested_state_dict([state_dict1, state_dict2], [0.75, 0.25])
    print(averaged_structured)
    
    # Verify the math manually for one tensor
    expected_weight = state_dict1["encoder"]["layer1"]["weight"] * 0.75 + state_dict2["encoder"]["layer1"]["weight"] * 0.25
    actual_weight = averaged_structured["encoder"]["layer1"]["weight"]
    
    print(f"\n=== Manual Verification ===")
    print(f"Expected encoder.layer1.weight: {expected_weight}")
    print(f"Actual encoder.layer1.weight: {actual_weight}")
    print(f"Match: {torch.allclose(expected_weight, actual_weight)}")
    
    print(f"\n=== Structure Comparison ===")
    print(f"Original has nested structure: {isinstance(state_dict1['encoder'], dict)}")
    print(f"Averaged has nested structure: {isinstance(averaged_structured['encoder'], dict)}")
