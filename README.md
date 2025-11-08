# Whisper Fine-Tuning & Aggregation Pipeline

This repository orchestrates aggregation and evaluation of federated Whisper checkpoints targeting Swiss German speech. The codebase now works directly with OpenAI's `whisper` package, streams Hugging Face datasets without local TSVs, and evaluates models end-to-end without subprocess calls.

## Highlights

- **OpenAI Whisper native workflow** – aggregate checkpoints and evaluate using `whisper.load_model` (`large-v3-turbo`).
- **Streaming aggregation** – average large state dicts in constant memory.
- **Streaming evaluation** – pull samples on-the-fly from `i4ds/spc_r` (or any HF dataset) without downloads.
- **W&B optional** – runs can log artifacts/metrics when `WANDB_API_KEY` is present, otherwise operate offline.
- **Single-process orchestration** – `run_aggregation_configs.py` iterates weight strategies and evaluates in-process.

## Key Scripts

- `model_aggregation.py` – streaming-friendly averaging utilities for Whisper `.pt` checkpoints.
- `average_runner.py` – CLI wrapper that aggregates checkpoints and saves `model.pt` containing only the state dict.
- `evaluate_model.py` – loads the base Whisper model, applies aggregated weights, and evaluates on a streaming HF dataset.
- `run_aggregation_configs.py` – drives a set of predefined aggregation strategies and evaluations.
- `audio_utils.py`, `whisper_utils.py` – helpers for audio preparation and fine-tuning (unchanged).

## Directory Layout

```
whisper-finetune-pipeline/
├── average_runner.py
├── evaluate_model.py
├── model_aggregation.py
├── run_aggregation_configs.py
├── audio_utils.py
├── whisper_utils.py
├── requirements.txt
├── checkpoints/
│   ├── wandering-river-4.pt
│   ├── smart-smoke-5.pt
│   ├── hearty-bee-7.pt
│   └── eager-haze-6.pt
└── whisper-aggregated/
    └── <strategy>/model.pt
```

Aggregated models are stored as plain PyTorch state dicts (`model.pt`). Loading always reconstructs the architecture from the official `whisper` package before weights are applied.

## Usage

### 1. Run predefined aggregation strategies

```
python run_aggregation_configs.py
```

The script checks for checkpoints, writes aggregated models under `whisper-aggregated/<strategy>/model.pt`, and evaluates each strategy on `i4ds/spc_r` (`test` split by default). CSV summaries (predictions + metrics) are produced per strategy.

### 2. Aggregate checkpoints manually

```
python average_runner.py \
    --checkpoints checkpoints/wandering-river-4.pt checkpoints/smart-smoke-5.pt \
    --weights 0.5 0.5 \
    --save_dir whisper-aggregated/custom
```

The output `whisper-aggregated/custom/model.pt` contains only the averaged state dict.

### 3. Evaluate a checkpoint directory

```
python evaluate_model.py \
    --model_dir whisper-aggregated/custom \
    --dataset_name i4ds/spc_r \
    --split test
```

The evaluator streams audio, transcribes with temperature 0.0, writes sample predictions to `evaluation_examples.csv`, and saves WER/BLEU scores to `evaluation_metrics.csv`.

## Checkpoint Format

- Saved files contain: `{tensor_name: tensor, ...}` (pure state dict).
- Evaluation always calls `whisper.load_model("large-v3-turbo")` and then loads the state dict.
- No additional metadata (`dims`, tokenizer files, etc.) is stored.

## Dataset & Evaluation Notes

- Default dataset: `i4ds/spc_r` (streaming). Override via CLI flags.
- Audio columns are cast to HF `Audio` with 16 kHz sampling on the fly.
- Evaluation keeps only a sample of examples (default 20) for CSV inspection.
- Metrics: WER (`evaluate.load("wer")`) and BLEU (`sacrebleu`).

## W&B Logging (Optional)

- Aggregation (`average_runner.py`): logs aggregated `model.pt` as an artifact when `WANDB_API_KEY` is configured.
- Evaluation (`evaluate_model.py`): logs metrics with contextual metadata if W&B credentials are available.
- Disable logging via `--disable_wandb` CLI flag.

## Requirements

- Python 3.9+
- `torch`, `torchaudio`, `numpy`, `evaluate`, `sacrebleu`, `datasets`, `whisper`, `wandb` (optional)
- Install with `pip install -r requirements.txt`

## Troubleshooting

- **Missing checkpoints** – `run_aggregation_configs.py` will skip aggregation and report missing files.
- **State dict mismatch** – ensure checkpoints were trained on `whisper-large-v3-turbo` before averaging.
- **Dataset access** – set `HF_HOME` / `HF_DATASETS_CACHE` if streaming fails; authentication may be required for private datasets.
- **OOM / crashes** – evaluation streams samples and avoids storing all predictions, but very long clips may still require more memory.

## License

MIT License. See `LICENSE` for details.
