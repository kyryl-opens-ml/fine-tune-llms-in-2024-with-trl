from dagster import Config, asset, MetadataValue, AssetExecutionContext
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
)
import os
import evaluate
from peft import AutoPeftModelForCausalLM
import torch
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from random import randint
from huggingface_hub import hf_hub_download
from collections import defaultdict
from tqdm import tqdm
import modal

HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")


class DataConfig(Config):
    dataset_name: str = "b-mc2/sql-create-context"
    train_data_path: str = "train_dataset-sql.json"
    test_data_path: str = "test_dataset-sql.json"
    test_size: float = 0.1
    sample_training: int = 5000


class ModelTrainingConfig(Config):
    pretrained_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_model_id: str = "text2sql-llama-3-8B"
    # mode: str = "modal-labs"  # or local


def create_conversation(sample):
    # Convert dataset to OAI messages
    system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
    SCHEMA:
    {schema}"""

    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(schema=sample["context"]),
            },
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


@asset(group_name="data", compute_kind="python")
def create_text_to_sql_dataset(config: DataConfig):
    if Path(config.train_data_path).exists() and Path(config.test_data_path).exists():
        return {
            "train_path": config.train_data_path,
            "test_path": config.test_data_path,
        }
    else:
        dataset = load_dataset(config.dataset_name, split="train")
        dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
        dataset = dataset.train_test_split(test_size=config.test_size)

        dataset["train"] = dataset["train"].shuffle().select(range(config.sample_training))
        # Save datasets to disk
        dataset["train"].to_json(config.train_data_path, orient="records")
        dataset["test"].to_json(config.test_data_path, orient="records")

        return {
            "train_path": config.train_data_path,
            "test_path": config.test_data_path,
        }


@asset(group_name="data", compute_kind="python")
def train_data(context: AssetExecutionContext, create_text_to_sql_dataset):
    dataset = load_dataset("json", data_files=create_text_to_sql_dataset["train_path"], split="train")

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset)),
            "sample": MetadataValue.json(dataset[randint(0, len(dataset))]),
        }
    )

    return dataset


@asset(group_name="data", compute_kind="python")
def test_data(context: AssetExecutionContext, create_text_to_sql_dataset):
    dataset = load_dataset("json", data_files=create_text_to_sql_dataset["test_path"], split="train")

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset)),
            "sample": MetadataValue.json(dataset[randint(0, len(dataset))]),
        }
    )

    return dataset


def run_training(pretrained_model_id: str, peft_model_id: str, train_data) -> str:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
    tokenizer.padding_side = "right"

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    # # LoRA config based on QLoRA paper & Sebastian Raschka experiment, but for speedup, we are going to use a more lightweight version of LoRA.
    # peft_config = LoraConfig(
    #     lora_alpha=128,
    #     lora_dropout=0.05,
    #     r=256,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )

    # We are going to train only q and v layer to speedup.
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=peft_model_id,  # directory to save and repository id
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=100,  # log every 10 steps
        save_steps=100,  # log every 10 steps
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=True,  # push model to hub
        report_to="none",  # report metrics to tensorboard
        hub_token=HF_TOKEN_WRITE,
    )

    max_seq_length = 3072  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    trainer.model.print_trainable_parameters()

    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_model()

    kwargs = {
        "finetuned_from": pretrained_model_id,
        "tasks": "text2sql",
        "language": "en",
    }
    trainer.create_model_card(**kwargs)

    hub_model_id = trainer.hub_model_id
    del trainer
    del model
    torch.cuda.empty_cache()

    return hub_model_id


@asset(group_name="model", compute_kind="modal-lab")
def trained_model(context: AssetExecutionContext, config: ModelTrainingConfig, train_data):
    run_training_modal_function = modal.Function.lookup("fine-tune-llms-in-2024-with-trl", "run_training_modal")
    hub_model_id = run_training_modal_function.remote(
        train_data_pandas=train_data.to_pandas(),
        pretrained_model_id=config.pretrained_model_id,
        peft_model_id=config.peft_model_id,
    )
    context.add_output_metadata({"model_url": MetadataValue.url(f"https://huggingface.co/{hub_model_id}")})
    return hub_model_id


@asset(group_name="model", compute_kind="python")
def model_card(context: AssetExecutionContext, trained_model):
    model_card_path = hf_hub_download(repo_id=trained_model, filename="README.md")
    with open(model_card_path, "r") as f:
        content = f.read()

    context.add_output_metadata({"content": MetadataValue.md(content)})
    return content


@asset(group_name="model", compute_kind="python")
def test_results(context: AssetExecutionContext, test_data, trained_model, config: ModelTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(trained_model)
    model = AutoPeftModelForCausalLM.from_pretrained(trained_model, device_map="auto", torch_dtype=torch.float16)

    merged_model = model.merge_and_unload()
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, torch_dtype=torch.float16)

    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    results = defaultdict(list)
    number_of_eval_samples = 10
    for s in tqdm(test_data.select(range(number_of_eval_samples))):
        query = s["messages"][1]["content"]
        prompt = pipe.tokenizer.apply_chat_template(s["messages"][:2], tokenize=False, add_generation_prompt=True)
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            eos_token_id=terminators,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        original_sql = s["messages"][2]["content"].lower()
        generated_sql = outputs[0]["generated_text"][len(prompt) :].strip().lower()

        results["query"].append(query)
        results["original_sql"].append(original_sql)
        results["generated_sql"].append(generated_sql)
        results["hard_match"].append(original_sql == generated_sql)

    rouge = evaluate.load("rouge")
    rouge_metrics = rouge.compute(predictions=results["generated_sql"], references=results["original_sql"])
    inference_samples = [
        {"original_sql": original_sql, "generated_sql": generated_sql, "hard_match": hard_match}
        for (original_sql, generated_sql, hard_match) in zip(
            results["original_sql"], results["generated_sql"], results["hard_match"]
        )
    ]

    context.add_output_metadata(
        {
            "inference_samples": MetadataValue.json(inference_samples),
            "rouge_metrics": MetadataValue.json({x: float(rouge_metrics[x]) for x in rouge_metrics}),
        }
    )
