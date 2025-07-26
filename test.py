import py_compile
from config_utils import generate_config
from typing import Dict 
py_compile.compile('config_utils.py', doraise=True)

if __name__ == "__main__":
    sample_config = {
        "training": {
            "epochs": 5,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "learning_rate": 3e-5,
            "optimizer": "adamw",
            "scheduler": "linear"
        },
        "model": {
            "base_model": "openai/whisper-small",
            "language": "de",
            "task": "transcribe",
            "use_fp16": True,
            "stochastic_depth": 0.1
        },
        "data": {
            "dataset_name": "mozilla-foundation/common_voice_13_0",
            "subset": "de",
            "streaming": False,
            "preprocessing": {
                "remove_background_noise": True,
                "max_duration": 30
            }
        },
        "logging": {
            "use_wandb": True,
            "wandb_project": "whisper-german-test",
            "log_every_n_steps": 10,
            "output_dir": "./checkpoints"
        }
    }
    generate_config("test_config.yaml", sample_config)
    print("Config file 'test_config.yaml' generated successfully.")