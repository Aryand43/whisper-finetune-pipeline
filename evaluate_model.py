import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoConfig
import re
import evaluate
import sacrebleu
import argparse
import csv
import wandb
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

def load_model_and_processor(model_dir: str):
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    config = AutoConfig.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())

def transcribe_dataset(model, processor, dataset):
    model.eval()
    predictions = []
    for sample in dataset:
        inputs = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        )
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
