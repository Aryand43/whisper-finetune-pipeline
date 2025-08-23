import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re 
import evaluate 
import sacrebleu
import argparse
import csv
import wandb
from datasets import load_dataset

def load_model_and_processor(model_path: str, base_model_name: str):
    processor = WhisperProcessor.from_pretrained(base_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    avg_state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(avg_state_dict)
    return model, processor
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path, args.base_model_name)
    dataset = load_dataset(args.dataset_name, split=args.split)
    predictions = transcribe_dataset(model, processor, dataset)
    references = [normalize_text(sample["text"]) for sample in dataset]
    predictions = [normalize_text(p) for p in predictions]
    metrics = evaluate_metrics(predictions, references)

    with open("results.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["wer", "bleu"])
        writer.writeheader()
        writer.writerow(metrics)

    wandb.init(project="whisper-eval")
    wandb.log(metrics)