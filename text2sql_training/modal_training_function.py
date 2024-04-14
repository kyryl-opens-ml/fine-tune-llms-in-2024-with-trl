import modal
from modal import Image
import pandas as pd
import os

app = modal.App("fine-tune-llms-in-2024-with-trl")
env = {"HF_TOKEN": os.getenv("HF_TOKEN"), "HF_TOKEN_WRITE": os.getenv("HF_TOKEN_WRITE")}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/fine-tune-llm-in-2024:main").env(env)


@app.function(
    image=custom_image,
    gpu="A100",
    mounts=[modal.Mount.from_local_python_packages("text2sql_training", "text2sql_training")],
    timeout=15 * 60,
)
def run_training_modal(train_data_pandas: pd.DataFrame, pretrained_model_id: str, peft_model_id: str):
    from datasets import Dataset
    from text2sql_training.llm_stf import run_training

    model_url = run_training(
        pretrained_model_id=pretrained_model_id,
        peft_model_id=peft_model_id,
        train_data=Dataset.from_pandas(train_data_pandas),
    )
    return model_url
