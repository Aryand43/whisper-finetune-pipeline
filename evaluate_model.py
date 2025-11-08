import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import evaluate
import sacrebleu
import torch
import wandb
from datasets import Audio, config as datasets_config, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

load_dotenv()
datasets_config.HF_DATASETS_AUDIO_BACKEND = "torchaudio"


def robust_hash_state_dict(state_dict: Dict[str, Any]) -> str:
    """Create a SHA256 hash for a (possibly nested) state dict."""

    def extract_tensor_bytes(node: Dict[str, Any]) -> List[bytes]:
        tensor_bytes: List[bytes] = []
        for key in sorted(node.keys()):
            value = node[key]
            if isinstance(value, dict):
                tensor_bytes.extend(extract_tensor_bytes(value))
            elif isinstance(value, torch.Tensor):
                tensor_bytes.append(value.detach().cpu().numpy().tobytes())
        return tensor_bytes

    flat_bytes = b"".join(extract_tensor_bytes(state_dict))
    return hashlib.sha256(flat_bytes).hexdigest()


def load_model_and_pipeline(model_dir: str, precision: str = "float16"):
    """Load Whisper model weights and return a ready-to-use ASR pipeline."""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = precision.lower()
    if precision not in {"float16", "fp32", "float32"}:
        raise ValueError(f"Unsupported precision: {precision}")

    torch_dtype = torch.float16 if (precision == "float16" and device != "cpu") else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.to(device)

    print(f"Loaded model from {model_dir}")
    print(f"Model weight SHA256 hash: {robust_hash_state_dict(model.state_dict())}")

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch_dtype if device != "cpu" else torch.float32,
        device=device,
    )


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for stable metrics."""

    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe_dataset(
    asr_pipe,
    dataset: Iterable[Dict[str, Any]],
    max_saved_examples: int = 20,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Stream the dataset, collect predictions, and keep the first few examples."""

    predictions: List[str] = []
    references: List[str] = []
    saved_examples: List[Dict[str, Any]] = []

    print("Transcribing samples (streaming)...")

    for idx, sample in enumerate(dataset):
        audio = sample["audio"]
        reference = sample.get("text") or sample.get("sentence")
        if reference is None:
            raise ValueError("Dataset sample is missing a reference transcript.")

        result = asr_pipe({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})
        prediction = result["text"]

        predictions.append(prediction)
        references.append(reference)

        if len(saved_examples) < max_saved_examples:
            saved_examples.append(
                {
                    "index": idx,
                    "prediction_raw": prediction,
                    "reference_raw": reference,
                    "prediction_normalized": normalize_text(prediction),
                    "reference_normalized": normalize_text(reference),
                }
            )

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} samples (streaming)")

    print(f"Finished transcription for {len(predictions)} samples")
    return predictions, references, saved_examples


def save_examples_to_csv(saved_examples: List[Dict[str, Any]], output_path: Optional[str]) -> None:
    if not output_path or not saved_examples:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "prediction_raw",
                "reference_raw",
                "prediction_normalized",
                "reference_normalized",
            ],
        )
        writer.writeheader()
        writer.writerows(saved_examples)

    print(f"Saved {len(saved_examples)} examples to {path}")


def write_metrics_to_csv(metrics: Dict[str, float], output_path: Optional[str]) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"Saved metrics to {path}")


def evaluate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    return {"wer": wer, "bleu": bleu}


def log_metrics_to_wandb(
    metrics: Dict[str, float],
    model_dir: str,
    dataset_name: str,
    split: str,
    precision: str,
    sample_count: int,
    run_name: Optional[str],
) -> None:
    if not os.getenv("WANDB_API_KEY"):
        return

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name,
        reinit=True,
    )
    wandb.log(
        {
            **metrics,
            "model_dir": model_dir,
            "dataset": dataset_name,
            "split": split,
            "precision": precision,
            "samples": sample_count,
        }
    )
    wandb.finish()


def evaluate_model(
    model_dir: str,
    dataset_name: str,
    split: str = "test",
    dataset_config: Optional[str] = None,
    precision: str = "float16",
    examples_csv: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    max_saved_examples: int = 20,
    log_to_wandb: bool = True,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a Whisper checkpoint on a streaming Hugging Face dataset."""

    asr_pipe = load_model_and_pipeline(model_dir, precision=precision)

    load_kwargs = {"split": split, "streaming": True}
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    predictions, references, saved_examples = transcribe_dataset(
        asr_pipe,
        dataset,
        max_saved_examples=max_saved_examples,
    )

    normalized_predictions = [normalize_text(p) for p in predictions]
    normalized_references = [normalize_text(r) for r in references]
    metrics = evaluate_metrics(normalized_predictions, normalized_references)

    save_examples_to_csv(saved_examples, examples_csv)
    write_metrics_to_csv(metrics, metrics_csv)

    if log_to_wandb:
        log_metrics_to_wandb(
            metrics,
            model_dir=model_dir,
            dataset_name=dataset_name,
            split=split,
            precision=precision,
            sample_count=len(predictions),
            run_name=wandb_run_name,
        )

    print(f"Evaluation metrics: {metrics}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--dataset_config", default=None, type=str)
    parser.add_argument("--precision", default="float16", type=str)
    parser.add_argument("--examples_csv", default="evaluation_examples.csv", type=str)
    parser.add_argument("--metrics_csv", default="evaluation_metrics.csv", type=str)
    parser.add_argument("--max_saved_examples", default=20, type=int)
    parser.add_argument("--wandb_run_name", default="evaluation_run", type=str)
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        dataset_config=args.dataset_config,
        precision=args.precision,
        examples_csv=args.examples_csv,
        metrics_csv=args.metrics_csv,
        max_saved_examples=args.max_saved_examples,
        log_to_wandb=not args.disable_wandb,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
