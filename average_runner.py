import argparse
import hashlib
import os
from typing import Optional, Sequence

import torch
import wandb
from transformers import WhisperForConditionalGeneration

from model_aggregation import average_checkpoints


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


def aggregate_models(
    checkpoints: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    *,
    save_dir: str = "whisper-aggregated",
    use_float64: bool = False,
    wandb_run_name: str = "model_aggregation_run",
    log_to_wandb: bool = True,
) -> str:
    """Average checkpoints and persist a Hugging Face-compatible Whisper model."""

    if not checkpoints:
        raise ValueError("No checkpoints provided for aggregation.")

    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)

    if len(weights) != len(checkpoints):
        raise ValueError("Weights must match checkpoints")

    run = None
    if log_to_wandb:
        run = wandb.init(
            entity="aryan-dutt43-whisper-federated",
            project="whisper-finetune-pipeline",
            job_type="model_aggregation",
            name=wandb_run_name,
            reinit=True,
        )

    print("Averaging checkpoints...")
    avg_state_dict = average_checkpoints(
        list(checkpoints),
        list(weights),
        use_float64=use_float64,
    )

    print(f"Hash of aggregated state dict: {robust_hash_state_dict(avg_state_dict)}")

    print("Loading base whisper-large-v3-turbo model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )

    print("Loading averaged weights into model...")
    base_model.load_state_dict(avg_state_dict, strict=True)

    print(f"Saving aggregated model to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    base_model.save_pretrained(save_dir)

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    processor.save_pretrained(save_dir)

    print(f"HF model and processor saved to: {save_dir}")
    print("Model aggregation completed successfully!")

    if run is not None:
        artifact = wandb.Artifact(name="aggregated_whisper_model", type="model")
        artifact.add_dir(save_dir)
        run.log_artifact(artifact)
        wandb.finish()

    return save_dir


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
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable logging aggregated model to Weights & Biases",
    )
    args = parser.parse_args()

    aggregate_models(
        args.checkpoints,
        args.weights,
        save_dir=args.save_dir,
        use_float64=args.use_float64,
        wandb_run_name=args.wandb_run_name,
        log_to_wandb=not args.disable_wandb,
    )


if __name__ == "__main__":
    main()
