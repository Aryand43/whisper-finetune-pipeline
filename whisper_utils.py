from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import pandas as pd
import torch
import torchaudio
import wandb
from datasets import Dataset,Audio
from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments
import evaluate

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

def train_whisper_model(model, processor, train_tsv_path: str, audio_dir: str, output_dir: str, num_epochs: int = 3, dry_run: bool = False, wandb_project: str = "whisper-default", wandb_entity: str = "default-entity", wandb_config: dict = None):
    wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config or {})
    df = pd.read_csv(train_tsv_path, sep="\t")
    df["audio"] = df["path"].apply(lambda x: os.path.join(audio_dir, x))
    ds = Dataset.from_pandas(df[["audio", "sentence"]])
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    audio_list = [x["array"] for x in ds["audio"]]
    sr_list = [x["sampling_rate"] for x in ds["audio"]]
    input_features = [processor(a, sampling_rate=sr, return_tensors="pt").input_features[0] for a, sr in zip(audio_list, sr_list)]
    with processor.as_target_processor():
        labels = [processor(t, return_tensors="pt").input_ids[0] for t in ds["sentence"]]
    ds = ds.remove_columns(["audio", "sentence"])
    ds = ds.add_column("input_features", input_features)
    ds = ds.add_column("labels", labels)
    wer = evaluate.load("wer")
    bleu = evaluate.load("sacrebleu")
    def_metrics = lambda pred: {
        "wer": wer.compute(
            predictions=processor.batch_decode(torch.argmax(torch.tensor(pred.predictions), dim=-1), skip_special_tokens=True),
            references=processor.batch_decode(pred.label_ids, skip_special_tokens=True)
        ),
        "bleu": bleu.compute(
            predictions=processor.batch_decode(torch.argmax(torch.tensor(pred.predictions), dim=-1), skip_special_tokens=True),
            references=processor.batch_decode(pred.label_ids, skip_special_tokens=True)
        )["score"]
    }
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        evaluation_strategy="no",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        logging_dir=os.path.join(output_dir, "logs"),
        predict_with_generate=True,
        report_to=["wandb"],
        save_total_limit=2
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=processor,
        compute_metrics=def_metrics
    )
    if dry_run:
        print("dry_run successful")
        return
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    load_whisper_model("medium", dry_run=True)