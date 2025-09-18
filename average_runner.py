import argparse
import torch
import os
from transformers import WhisperForConditionalGeneration

def flatten_state_dict(state, parent_key='', sep='.'):
    items = {}
    for k, v in state.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_state_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def average_checkpoints(model_paths, weights=None, save_path="whisper-aggregated/pytorch_model.bin"):
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    assert len(weights) == len(model_paths), "Mismatch between number of models and weights"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    avg_state_dict = None
    for path, weight in zip(model_paths, weights):
        loaded = torch.load(path, map_location="cpu")

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(avg_state_dict, save_path)
    print(f"Averaged model saved at: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of .pt model paths")
    parser.add_argument("--weights", nargs="+", type=float, help="Optional list of weights")
    parser.add_argument("--save_dir", default="whisper-aggregated", help="Directory to save HuggingFace-compatible model")
    args = parser.parse_args()

    if args.weights:
        assert len(args.weights) == len(args.checkpoints), "Weights must match checkpoints"
        weights = args.weights
    else:
        weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)

    save_path = os.path.join(args.save_dir, "pytorch_model.bin")
    average_checkpoints(args.checkpoints, weights, save_path)

    # Save HF wrapper
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    avg_weights = torch.load(save_path)
    base_model.load_state_dict(avg_weights)  # avoid crash if some keys mismatch
    base_model.save_pretrained(args.save_dir)

    print(f"HF model saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
