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
    print("Based on real whisper-large-v3-turbo structure analysis...\n")
    
    # Create two mock state dictionaries that match the REAL whisper structure
    # Keys are already "flattened" like in the actual model: model.encoder.conv1.weight
    state_dict1 = {
        "model.encoder.conv1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "model.encoder.conv1.bias": torch.tensor([0.5, 1.5]),
        "model.encoder.layer_norm.weight": torch.tensor([1.0, 1.0]),
        "model.encoder.layer_norm.bias": torch.tensor([0.1, 0.2]),
        "model.decoder.embed_tokens.weight": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "model.decoder.layer_norm.weight": torch.tensor([2.0, 2.0]),
        "proj_out.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    }
    
    state_dict2 = {
        "model.encoder.conv1.weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        "model.encoder.conv1.bias": torch.tensor([1.0, 2.0]),
        "model.encoder.layer_norm.weight": torch.tensor([1.5, 1.5]),
        "model.encoder.layer_norm.bias": torch.tensor([0.2, 0.3]),
        "model.decoder.embed_tokens.weight": torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
        "model.decoder.layer_norm.weight": torch.tensor([2.5, 2.5]),
        "proj_out.weight": torch.tensor([[8.0, 9.0], [10.0, 11.0]])
    }
    
    print("=== Mock Checkpoint 1 (simulates training checkpoint) ===")
    checkpoint1 = {
        "state_dict": {
            "model": state_dict1
        },
        "epoch": 5,
        "lr": 1e-5
    }
    print(f"Checkpoint keys: {list(checkpoint1.keys())}")
    print(f"Model state dict keys (first 3): {list(state_dict1.keys())[:3]}")
    
    print("\n=== Mock Checkpoint 2 ===")
    checkpoint2 = {
        "state_dict": {
            "model": state_dict2
        },
        "epoch": 10,
        "lr": 5e-6
    }
    print(f"Checkpoint keys: {list(checkpoint2.keys())}")
    print(f"Model state dict keys (first 3): {list(state_dict2.keys())[:3]}")
    
    # Create temporary checkpoint files to test the full pipeline
    print("\n=== Testing Full Checkpoint Loading Pipeline ===")
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    checkpoint1_path = os.path.join(temp_dir, "checkpoint1.pt")
    checkpoint2_path = os.path.join(temp_dir, "checkpoint2.pt")
    
    torch.save(checkpoint1, checkpoint1_path)
    torch.save(checkpoint2, checkpoint2_path)
    
    print(f"Created temporary checkpoints:")
    print(f"  - {checkpoint1_path}")
    print(f"  - {checkpoint2_path}")
    
    # Test our extraction and aggregation
    try:
        print("\n=== Testing Checkpoint Aggregation ===")
        aggregated_result = average_checkpoints_structured(
            [checkpoint1_path, checkpoint2_path], 
            [0.7, 0.3],
            os.path.join(temp_dir, "aggregated.pt")
        )
        
        print("✅ Aggregation successful!")
        print(f"Aggregated result keys (first 5): {list(aggregated_result.keys())[:5]}")
        
        # Verify the math for one tensor
        expected_weight = state_dict1["model.encoder.conv1.weight"] * 0.7 + state_dict2["model.encoder.conv1.weight"] * 0.3
        actual_weight = aggregated_result["model.encoder.conv1.weight"]
        
        print(f"\n=== Manual Verification ===")
        print(f"Expected model.encoder.conv1.weight: {expected_weight}")
        print(f"Actual model.encoder.conv1.weight: {actual_weight}")
        print(f"Match: {torch.allclose(expected_weight, actual_weight)}")
        
        # Test that this would work with the real model loading
        print(f"\n=== Compatibility Test ===")
        print(f"Result has whisper-like keys: {'model.encoder.conv1.weight' in aggregated_result}")
        print(f"Result has proj_out: {'proj_out.weight' in aggregated_result}")
        print(f"All keys are strings: {all(isinstance(k, str) for k in aggregated_result.keys())}")
        print(f"All values are tensors: {all(isinstance(v, torch.Tensor) for v in aggregated_result.values())}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary files")
    
    print(f"\n=== Key Insights ===")
    print("1. Real Whisper models already have 'flattened' keys like 'model.encoder.conv1.weight'")
    print("2. Training checkpoints wrap this in: checkpoint['state_dict']['model']")
    print("3. Our aggregation preserves the exact key structure needed for model.load_state_dict()")
    print("4. No additional flattening/unflattening needed - keys match original model!")
