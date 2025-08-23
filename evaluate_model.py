import os
from dotenv import load_dotenv

load_dotenv()  # Load env vars from .env file

import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY")
)