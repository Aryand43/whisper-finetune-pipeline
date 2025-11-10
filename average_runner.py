import argparse
import hashlib
import os
from typing import Any, Optional, Sequence

import torch
import wandb
import whisper

from model_aggregation import average_checkpoints


def robust_hash_state_dict(state_dict):
    def extract_tensor_bytes(d):
        tensor_bytes = []
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                tensor_bytes.extend(extract_tensor_bytes(value))
            elif isinstance(value, torch.Tensor):
                tensor_bytes.append(value.detach().cpu().numpy().tobytes())
        return tensor_bytes

    all_bytes = b"".join(extract_tensor_bytes(state_dict))
    return hashlib.sha256(all_bytes).hexdigest()


def aggregate_models(
    checkpoints: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    *,
    save_dir: str = "whisper-aggregated",
    use_float64: bool = False,
    wandb_run: Optional[Any] = None,
) -> str:
    """Average checkpoints and persist an OpenAI Whisper model checkpoint."""

    if not checkpoints:
        raise ValueError("No checkpoints provided for aggregation.")

    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)

    if len(weights) != len(checkpoints):
        raise ValueError("Weights must match checkpoints")

    print("Averaging checkpoints...")
    avg_state_dict = average_checkpoints(
        list(checkpoints),
        list(weights),
        use_float64=use_float64,
    )

    print(f"Hash of aggregated state dict: {robust_hash_state_dict(avg_state_dict)}")

    print("Loading base whisper-large-v3-turbo model (OpenAI)...")
    base_model = whisper.load_model("large-v3-turbo", device="cpu")

    print("Loading averaged weights into model...")
    base_model.load_state_dict(avg_state_dict, strict=True)

    print(f"Saving aggregated model to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "model.pt")
    torch.save(base_model.state_dict(), checkpoint_path)

    print(f"OpenAI Whisper checkpoint saved to: {checkpoint_path}")
    print("Model aggregation completed successfully!")

    if wandb_run is not None:
        artifact = wandb.Artifact(name="aggregated_whisper_model", type="model")
        artifact.add_file(checkpoint_path)
        wandb_run.log_artifact(artifact)

    return checkpoint_path


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
    args = parser.parse_args()

    run = wandb.init(
        entity="aryan-dutt43-whisper-federated",
        project="whisper-finetune-pipeline",
        job_type="model_aggregation",
        name="model_aggregation_run",
        reinit=True,
    )

    aggregate_models(
        args.checkpoints,
        args.weights,
        save_dir=args.save_dir,
        use_float64=args.use_float64,
        wandb_run=run,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
