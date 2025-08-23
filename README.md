# Whisper Fine-Tuning and Aggregation for Swiss German

This repository fine-tunes OpenAI's Whisper model on Swiss German using synthetic long-form audio. It also supports model aggregation, evaluation on public corpora, and low-resource quantization options.

## Features

- Synthetic long-form audio generation via VAD and overlap stitching
- Timestamp-preserving segmentation
- Aggregation of multiple fine-tuned model checkpoints
- Evaluation on Swiss Parliament Corpus Re‑Imagined (SPC-R)
- Support for float16 and int8 quantized inference
- WER and BLEU computation with Weights & Biases logging

## Components

- `audio_utils.py`: Converts `.flac` clips to long-form `.wav` with consistent SHA256 hashes
- `whisper_utils.py`: Fine-tunes Whisper using TSV files and audio directories
- `model_aggregation.py`: Averages multiple `best_model.pt` files and evaluates the aggregated model
- `average_runner.py`: Script to run model aggregation with custom weight distribution
- `requirements.txt`: Includes core libraries and optional quantization dependencies

## Directory Structure

```
aryand43-whisper-finetune-pipeline/
├── README.md
├── audio_utils.py
├── average_runner.py
├── model_aggregation.py
├── whisper_utils.py
├── requirements.txt
├── whisper-aggregated/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── other Hugging Face config files
└── checkpoints/
    ├── hearty-bee-7.pt
    ├── smart-smoke-5.pt
    ├── eager-haze-6.pt
    └── wandering-river-4.pt
```

## Requirements

- Python 3.8+
- transformers >= 4.41.1
- torch >= 2.3.0
- torchaudio
- pydub >= 0.25.1
- numpy < 2.0
- pandas
- evaluate
- sacrebleu
- wandb
- bitsandbytes (optional for quantization)
- accelerate (optional for quantization)
