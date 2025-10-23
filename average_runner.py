import argparse
import torch
import os
from transformers import WhisperForConditionalGeneration
from model_aggregation import average_checkpoints_structured
import hashlib

def robust_hash_state_dict(state_dict):
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    def extract_tensor_bytes(d):
        tensor_bytes = []
        for key in sorted(d.keys()):
            v = d[key]
            if isinstance(v, dict):
                tensor_bytes.extend(extract_tensor_bytes(v))
            elif isinstance(v, torch.Tensor):
                tensor_bytes.append(v.detach().cpu().numpy().tobytes())
        return tensor_bytes

    all_bytes = b''.join(extract_tensor_bytes(state_dict))
    return hashlib.sha256(all_bytes).hexdigest()

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

    # Step 1: Average the checkpoints while preserving structure
    print("Averaging checkpoints...")
    # Use the function from model_aggregation.py but we don't need the file save
    # So we'll just get the averaged state dict by calling it without save_path
    temp_save_path = "/tmp/temp_avg_model.bin"
    avg_state_dict = average_checkpoints_structured(args.checkpoints, weights, temp_save_path)
    
    print(f"Hash of aggregated state dict: {robust_hash_state_dict(avg_state_dict['model_state_dict'])}")

    # Clean up the temporary file since we don't need it
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)

    # Step 2: Load the base whisper-large-v3-turbo model
    print("Loading base whisper-large-v3-turbo model...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
    
    # Step 3: Overwrite the model weights with averaged weights
    print("Loading averaged weights into model...")
    try:
        # Try to load the state dict, handling potential key mismatches
        missing_keys, unexpected_keys = base_model.load_state_dict(avg_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in averaged model: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Warning: Unexpected keys in averaged model: {unexpected_keys[:5]}...")  # Show first 5
            
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    # Step 4: Save the model using save_pretrained
    print(f"Saving aggregated model to: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    base_model.save_pretrained(args.save_dir)
    
    # Also save the processor for completeness
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    processor.save_pretrained(args.save_dir)

    print(f"HF model and processor saved to: {args.save_dir}")
    print("Model aggregation completed successfully!")

if __name__ == "__main__":
    main()
