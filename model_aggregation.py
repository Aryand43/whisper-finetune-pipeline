import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
import evaluate
import sacrebleu
import argparse
import csv
import wandb
from datasets import load_dataset
import os
from dotenv import load_dotenv
from transformers import AutoConfig
load_dotenv()  # Load env vars from .env file

import wandb
base_model_name = "openai/whisper-medium"
dataset = load_dataset("i4ds/spc_r", split="test")
# -----------------------------
# Model Loader (HF-style folder)
# -----------------------------
def load_model_and_processor(model_dir: str):
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    
    config = AutoConfig.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration(config)
    model.load_state_dict(torch.load(os.path.join(model_dir, "aggregated_model.bin")))
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

# -----------------------------
# Checkpoint Averaging Function
# -----------------------------
def average_checkpoints(model_paths, weights=None, save_path="aggregated_model.bin"):
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    assert len(weights) == len(model_paths), "Weights must match number of models"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    avg_state_dict = None

    for model_path, weight in zip(model_paths, weights):
        loaded = torch.load(model_path, map_location="cpu")
        state_dict = loaded["state_dict"]["model"] if "state_dict" in loaded and "model" in loaded["state_dict"] else loaded

        for k, v in state_dict.items():
            avg_state_dict[k] = v.clone().float() * weight
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() * weight for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float() * weight

    torch.save(avg_state_dict, save_path)
    print(f"Averaged model saved at: {save_path}")

# -----------------------------
# Evaluation Utilities
# -----------------------------
def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())

def transcribe_dataset(model, processor, dataset):
    model.eval()
    predictions = []
    for sample in dataset:
        inputs = processor(sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
    return predictions

def evaluate_metrics(predictions, references):
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    return {"wer": wer, "bleu": bleu}

# -----------------------------
# Main Evaluation Script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of HF-style model folder")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_dir)
    dataset = load_dataset(args.dataset_name, split=args.split)

    predictions = transcribe_dataset(model, processor, dataset)
    references = [normalize_text(sample["text"]) for sample in dataset]
    predictions = [normalize_text(p) for p in predictions]
    metrics = evaluate_metrics(predictions, references)

    with open("results.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["wer", "bleu"])
        writer.writeheader()
        writer.writerow(metrics)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name="aggregated-model"
    )
    wandb.log(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
