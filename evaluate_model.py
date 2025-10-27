import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
import evaluate
import sacrebleu
import argparse
import csv
import wandb
from datasets import load_dataset, config as datasets_config
import os
from dotenv import load_dotenv
import random
import hashlib

# Load env variables
load_dotenv()
datasets_config.HF_DATASETS_AUDIO_BACKEND = "torchaudio"

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

def load_model_and_pipeline(model_dir: str, precision: str = "float16"):
    """Load model and create pipeline with chunking support."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() and precision == "float16" else torch.float32
    
    # Try to load from the model directory first, fallback to base model
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_dir)
        print(f"Loaded model and processor from model directory: {model_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}. Check if the directory exists and has the correct files.\nError: {e}")
    model.to(device)
    print(f"Model weight SHA256 hash: {robust_hash_state_dict(model.state_dict())}")

    # Create pipeline with chunking support
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe, processor

def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())

def transcribe_dataset(pipe, dataset, save_probability):
    """Transcribe dataset using pipeline with 30-second chunking support."""
    predictions = []
    references = []
    saved_examples = []
    total_samples = len(dataset)
    print(f"Transcribing {total_samples} samples with 30-second chunking...")
    print(f"Saving examples with {save_probability*100:.1f}% probability...")

    for i, sample in enumerate(dataset):
        # Use the pipeline for transcription with automatic chunking
        audio_data = {
            "array": sample["audio"]["array"],
            "sampling_rate": sample["audio"]["sampling_rate"]
        }
        
        result = pipe(audio_data)
        transcription = result["text"]
        
        predictions.append(transcription)
        references.append(sample["sentence"] if "sentence" in sample else sample["text"])
        
        # Randomly decide whether to save this example
        if random.random() < save_probability:
            saved_examples.append({
                'index': i,
                'prediction': transcription,
                'reference': sample["sentence"] if "sentence" in sample else sample["text"],
            })
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_samples} samples (saved {len(saved_examples)} examples so far)")

    print(f"📊 Final: Processed {len(predictions)} samples, saved {len(saved_examples)} examples ({len(saved_examples)/len(predictions)*100:.1f}%)")
    return predictions, references, saved_examples

def save_examples_to_csv(saved_examples, output_path="evaluation_examples.csv"):
    """
    Save randomly selected evaluation examples to CSV with both normalized and non-normalized versions.
    
    Args:
        saved_examples: List of dictionaries with 'index', 'prediction', 'reference' keys
        output_path: Path to save the CSV file
    """
    if not saved_examples:
        print("⚠️  No examples to save!")
        return []
    
    examples = []
    for i, ex in enumerate(saved_examples):
        pred_raw = ex['prediction']
        ref_raw = ex['reference']
        pred_normalized = normalize_text(pred_raw)
        ref_normalized = normalize_text(ref_raw)
        
        examples.append({
            "sample_index": ex['index'],
            "example_id": i + 1,
            "reference_raw": ref_raw,
            "prediction_raw": pred_raw,
            "reference_normalized": ref_normalized,
            "prediction_normalized": pred_normalized,
        })
    
    # Save to CSV
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_index",
            "example_id", 
            "reference_raw", 
            "prediction_raw",
            "reference_normalized", 
            "prediction_normalized",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    
    print(f"📄 Saved {len(examples)} examples to: {output_path}")
    
    return examples

def evaluate_metrics(predictions, references):
    """Compute WER and BLEU metrics on normalized text."""
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    return {"wer": wer, "bleu": bleu}

import torchaudio

def force_decode_with_torchaudio(example):
    audio_array, sampling_rate = example["audio"]["array"], example["audio"]["sampling_rate"]
    return {
        "audio_array": audio_array,
        "sampling_rate": sampling_rate,
        "transcription": example["sentence"] if "sentence" in example else example["text"],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of HF-style model folder")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, help="Dataset configuration (e.g., 'gsw' for Swiss German)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--precision", type=str, default="float16", choices=["fp32", "float16", "int8"])
    parser.add_argument("--examples_csv", type=str, default="evaluation_examples.csv", help="Path to save example CSV")
    parser.add_argument("--metrics_csv", type=str, default="evaluation_metrics.csv", help="Path to save metrics CSV")
    args = parser.parse_args()

    print(f"🚀 Loading model from: {args.model_dir}")
    pipe, processor = load_model_and_pipeline(args.model_dir, precision=args.precision)

    print(f"📚 Loading dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset_name, split=args.split)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Removed dataset.map(force_decode_with_torchaudio) as it caused std::bad_alloc
    # The pipeline should handle audio loading from the 'audio' column directly.
    # dataset = dataset.map(force_decode_with_torchaudio)
    dataset = dataset.with_format("python")

    print(f"🎯 Starting transcription...")
    predictions, references, saved_examples = transcribe_dataset(pipe, dataset, save_probability=1)
    # Save examples to CSV (before normalization)
    print(f"💾 Saving examples...")
    save_examples_to_csv(saved_examples, args.examples_csv)
    
    # Normalize for metrics computation
    print(f"📊 Computing metrics...")
    references_normalized = [normalize_text(ref) for ref in references]
    predictions_normalized = [normalize_text(pred) for pred in predictions]
    metrics = evaluate_metrics(predictions_normalized, references_normalized)
    
    print(f"\n🎯 Results:")
    print(f"  WER: {metrics['wer']:.4f}")
    print(f"  BLEU: {metrics['bleu']:.2f}")

    # Save metrics to CSV
    metrics_file = args.metrics_csv
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["wer", "bleu", "model_dir", "dataset", "samples"])
        writer.writeheader()
        writer.writerow({
            "wer": metrics["wer"],
            "bleu": metrics["bleu"], 
            "model_dir": args.model_dir,
            "dataset": f"{args.dataset_name}:{args.dataset_config}" if args.dataset_config else args.dataset_name,
            "samples": len(predictions)
        })
    print(f"📊 Saved metrics to: {metrics_file}")

    # Log to wandb if configured
    if os.getenv("WANDB_API_KEY"):
        print(f"📡 Logging to wandb...")
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"eval-{args.model_dir.replace('/', '-')}-{args.precision}"
        )
        wandb.log({
            **metrics,
            "model_dir": args.model_dir,
            "dataset": args.dataset_name,
            "samples": len(predictions),
            "precision": args.precision
        })
        wandb.finish()
        print(f"✅ Logged to wandb")
    else:
        print(f"⚠️  Skipping wandb logging (no API key found)")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   📄 Examples: {args.examples_csv}")
    print(f"   📊 Metrics: {metrics_file}")

if __name__ == "__main__":
    main()
