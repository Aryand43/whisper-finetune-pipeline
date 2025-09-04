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

def load_model_and_processor(model_dir: str, precision: str = "fp32"):
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

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

def force_decode_with_torchaudio(example):
    audio_path = example["audio"]["path"]
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    example["audio"]["array"] = waveform.squeeze().numpy()
    example["audio"]["sampling_rate"] = 16000
    return example

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
