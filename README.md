# Whisper Fine-Tuning for Swiss German

Fine-tunes OpenAI's Whisper model on Swiss German using synthetic long-form audio created from short utterances. Aims to improve real-world transcription with timestamp accuracy and dialect generalization.

## Features

* Synthetic long-form audio generation via VAD + overlap stitching
* Preserves segmentation (timestamps)
* Generalizes to real-world audio
* BLEU & SubER improvements

## Functions

* `generate_config()`: Auto-generates YAML configs
* `generate_longform_audio()`: Stitches short clips into coherent long-form audio

## Directory

```
aryand43-whisper-finetune-pipeline/
├── README.md
├── audio_utils.py
├── config_utils.py
├── requirements.txt
├── test.py
└── test_config.yaml
```

## Requirements

* Python 3.8+
* webrtcvad
* pydub
* pandas
* PyYAML
