import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoConfig
import re
import evaluate
import sacrebleu
import argparse
import csv
import wandb
from datasets import load_dataset, config as datasets_config
import os
from dotenv import load_dotenv
import torchaudio

# Load env variables
load_dotenv()
datasets_config.HF_DATASETS_AUDIO_BACKEND = "torchaudio"

def load_model_and_processor(model_dir: str, precision: str = "float16"):
    # Try to load processor from the model directory first, fallback to base model
    try:
        processor = WhisperProcessor.from_pretrained(model_dir)
        print(f"Loaded processor from model directory: {model_dir}")
    except Exception as e:
        print(f"Could not load processor from {model_dir}, using whisper-large-v3-turbo as fallback")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

    if precision == "float16":
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto"
        )
    elif precision == "int8":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, quantization_config=quant_config, device_map="auto"
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    return model, processor  # Remove the manual `.to(...)` line

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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append(transcription)

    return predictions

def evaluate_metrics(predictions, references):
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
        "transcription": example["text"],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of HF-style model folder")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "float16", "int8"])
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_dir, precision=args.precision)

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.map(force_decode_with_torchaudio)
    dataset = dataset.with_format("python")

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
        name=f"aggregated-model-{args.precision}"
    )
    wandb.log(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
