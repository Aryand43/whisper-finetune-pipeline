from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def load_whisper_model(model_size: str, dry_run: bool = False):
    if dry_run:
        print(f"[DRY RUN] Would load: openai/whisper-{model_size}")
        return None, None
    model_path = f"openai/whisper-{model_size}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    except Exception as e:
        raise ValueError(f"Failed to load Whisper model '{model_path}'. Check model size. Error: {e}")

    return model, processor

if __name__ == "__main__":
    load_whisper_model("medium", dry_run=True)