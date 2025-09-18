# Whisper Fine-Tuning and Aggregation for Swiss German

This repository fine-tunes OpenAI's Whisper model on Swiss German using synthetic long-form audio. It supports model aggregation from local checkpoints or directly from Weights & Biases (wandb), evaluation on public corpora, and low-resource quantization options.

## Features

- **Structure-preserving model aggregation**: Maintains nested state dict structure for proper model loading
- **Wandb integration**: Download and aggregate models directly from wandb runs
- **Whisper-large-v3-turbo support**: Uses the latest Whisper model as base
- Synthetic long-form audio generation via VAD and overlap stitching
- Timestamp-preserving segmentation
- Evaluation on Swiss Parliament Corpus Re‑Imagined (SPC-R)
- Support for float16 and int8 quantized inference
- WER and BLEU computation with Weights & Biases logging

## Components

- `audio_utils.py`: Converts `.flac` clips to long-form `.wav` with consistent SHA256 hashes
- `whisper_utils.py`: Fine-tunes Whisper using TSV files and audio directories
- `model_aggregation.py`: **[UPDATED]** Structure-preserving averaging of multiple checkpoints
- `average_runner.py`: Script to run model aggregation with custom weight distribution (local files)
- `wandb_aggregator.py`: **[NEW]** Download and aggregate models directly from wandb runs
- `evaluate_model.py`: **[UPDATED]** Flexible evaluation supporting aggregated models
- `requirements.txt`: Includes core libraries and optional quantization dependencies

## Directory Structure

```
whisper-finetune-pipeline/
├── README.md
├── audio_utils.py
├── average_runner.py                 # Local checkpoint aggregation
├── wandb_aggregator.py              # Wandb checkpoint aggregation [NEW]
├── model_aggregation.py             # Core aggregation functions [UPDATED]
├── evaluate_model.py                # Model evaluation [UPDATED]
├── whisper_utils.py
├── requirements.txt
├── whisper-aggregated/              # Output directory
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── other HuggingFace config files
└── checkpoints/                     # Local checkpoints (optional)
    ├── run1_last_model.pt
    ├── run2_last_model.pt
    └── run3_last_model.pt
```

## Quick Start

### Option 1: Aggregate from Wandb Runs (Recommended)

Download and aggregate models directly from wandb:

```bash
# Using wandb run paths
python wandb_aggregator.py \
    --wandb-runs entity/project/run1 entity/project/run2 entity/project/run3 \
    --weights 0.4 0.3 0.3 \
    --save-dir whisper-aggregated

# Using full wandb URLs  
python wandb_aggregator.py \
    --wandb-urls \
        "https://wandb.ai/aryan-dutt43-whisper-federated/whisper-federated/runs/yg5oaq50/files/55089672_output" \
        "https://wandb.ai/aryan-dutt43-whisper-federated/whisper-federated/runs/abc123/files/output" \
    --save-dir whisper-aggregated

# Equal weights (default)
python wandb_aggregator.py \
    --wandb-runs entity/project/run1 entity/project/run2 \
    --save-dir whisper-aggregated
```

### Option 2: Aggregate from Local Checkpoints

```bash
python average_runner.py \
    --checkpoints checkpoints/run1_last_model.pt checkpoints/run2_last_model.pt \
    --weights 0.6 0.4 \
    --save-dir whisper-aggregated
```

### Option 3: Evaluate Aggregated Model

```bash
python evaluate_model.py \
    --model-dir whisper-aggregated \
    --dataset-name mozilla-foundation/common_voice_17_0 \
    --dataset-config gsw \
    --precision fp32
```

## Advanced Usage

### Wandb Integration Details

The `wandb_aggregator.py` script supports multiple input formats:

1. **Run paths**: `entity/project/run_id`
   ```bash
   python wandb_aggregator.py --wandb-runs aryan-dutt43-whisper-federated/whisper-federated/yg5oaq50
   ```

2. **Full URLs**: Complete wandb URLs
   ```bash
   python wandb_aggregator.py --wandb-urls "https://wandb.ai/entity/project/runs/run_id/files/..."
   ```

3. **Custom checkpoint names**: If your checkpoints aren't named `last_model.pt`
   ```bash
   python wandb_aggregator.py \
       --wandb-runs run1 run2 run3 \
       --checkpoint-name best_model.pt
   ```

### Structure-Preserving Aggregation

The aggregation now preserves the nested structure of PyTorch state dictionaries:

- ✅ **Before**: `{"encoder": {"layer1": {"weight": tensor}}}`
- ❌ **Old approach**: `{"encoder.layer1.weight": tensor}` (flattened - breaks loading)
- ✅ **New approach**: `{"encoder": {"layer1": {"weight": tensor}}}` (preserved structure)

This ensures the aggregated model can be properly loaded into the original architecture.

### Model Compatibility

- **Base model**: `openai/whisper-large-v3-turbo`
- **Input checkpoints**: Any Whisper fine-tuned checkpoints with compatible architecture
- **Output format**: HuggingFace compatible model directory with `save_pretrained()`

## API Reference

### Core Functions

```python
from model_aggregation import average_nested_state_dict, average_checkpoints_structured
from wandb_aggregator import aggregate_wandb_checkpoints

# Structure-preserving averaging
avg_state_dict = average_nested_state_dict(state_dicts, weights)

# Direct wandb aggregation
save_dir = aggregate_wandb_checkpoints(
    wandb_sources=["entity/project/run1", "entity/project/run2"],
    weights=[0.6, 0.4],
    save_dir="output",
    checkpoint_name="last_model.pt"
)
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

## Installation

```bash
pip install -r requirements.txt
wandb login  # Required for wandb integration
```

## Examples

### Real-world Example with Wandb URLs

```bash
# Aggregate 3 models from wandb with custom weights
python wandb_aggregator.py \
    --wandb-urls \
        "https://wandb.ai/aryan-dutt43-whisper-federated/whisper-federated/runs/yg5oaq50/files/55089672_output" \
        "https://wandb.ai/aryan-dutt43-whisper-federated/whisper-federated/runs/abc123/files/output" \
        "https://wandb.ai/aryan-dutt43-whisper-federated/whisper-federated/runs/def456/files/output" \
    --weights 0.5 0.3 0.2 \
    --save-dir ./final-whisper-model \
    --checkpoint-name last_model.pt

# Evaluate the aggregated model
python evaluate_model.py \
    --model-dir ./final-whisper-model \
    --dataset-name mozilla-foundation/common_voice_17_0 \
    --dataset-config gsw
```

## Troubleshooting

### Common Issues

1. **Wandb authentication**: Run `wandb login` before using wandb features
2. **Missing checkpoints**: Ensure checkpoint name matches (default: `last_model.pt`)
3. **Model loading errors**: Structure-preserving aggregation should resolve this
4. **Memory issues**: Use `--precision float16` or `int8` for evaluation

### Error Messages

- `"Checkpoint 'last_model.pt' not found"`: Check checkpoint name with `--checkpoint-name`
- `"Could not parse wandb URL"`: Ensure URL format is correct
- `"Missing keys in averaged model"`: Normal warning, indicates some layers weren't in checkpoints

## Contributing

This pipeline uses structure-preserving aggregation to maintain model compatibility. When contributing:

1. Preserve nested dictionary structures
2. Test with real wandb checkpoints
3. Ensure HuggingFace compatibility with `save_pretrained()`

## License

MIT License - see LICENSE file for details.
