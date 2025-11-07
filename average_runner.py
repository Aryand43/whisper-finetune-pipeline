import argparse
import torch
import os
from transformers import WhisperForConditionalGeneration
from model_aggregation import average_checkpoints
import hashlib
import wandb  # Import wandb


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

    all_bytes = b"".join(extract_tensor_bytes(state_dict))
    return hashlib.sha256(all_bytes).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="List of .pt model paths"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, help="Optional list of weights"
    )
    parser.add_argument(
        "--save_dir",
        default="whisper-aggregated",
        help="Directory to save HuggingFace-compatible model",
    )
    parser.add_argument(
        "--use_float64",
        action="store_true",
        help="Use float64 for accumulation during averaging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="model_aggregation_run",
        help="Name for the Weights & Biases run",
    )
    args = parser.parse_args()

    # Initialize Weights & Biases run
    run = wandb.init(
        entity="aryan-dutt43-whisper-federated",
        project="whisper-finetune-pipeline",
        job_type="model_aggregation",
        name=args.wandb_run_name,
        reinit=True,
    )

    if args.weights:
        assert len(args.weights) == len(
            args.checkpoints
        ), "Weights must match checkpoints"
        weights = args.weights
    else:
        weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)

    # Step 1: Average the checkpoints while preserving structure
    print("Averaging checkpoints...")
    avg_state_dict = average_checkpoints(
        args.checkpoints,
        weights,
        use_float64=args.use_float64,
    )

    print(f"Hash of aggregated state dict: {robust_hash_state_dict(avg_state_dict)}")

    # Step 2: Load the base whisper-large-v3-turbo model
    print("Loading base whisper-large-v3-turbo model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )

    # Step 3: Overwrite the model weights with averaged weights
    print("Loading averaged weights into model...")

    # Try to load the state dict, handling potential key mismatches
    _, _ = base_model.load_state_dict(avg_state_dict, strict=True)

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

    # Log the saved model directory as a W&B artifact
    artifact = wandb.Artifact(name="aggregated_whisper_model", type="model")
    artifact.add_dir(args.save_dir)
    run.log_artifact(artifact)

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
