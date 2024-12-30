# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "rich",
#     "transformers",
#     "python-dotenv",
#     "datasets>=2.14.6",
#     "accelerate",
#     "requests",
#     "cryptography",
#     "logging"
# ]
# ///
# train.py

import os
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv
import logging
from logging_setup import setup_logging, get_logger
from initialize_project import initialize_project
from deploy_runpod import automate_workflow
import requests
from cryptography.fernet import Fernet
import json
from typing import Optional

load_dotenv()

def encrypt_env(config: dict) -> tuple[bytes, bytes]:
    """Encrypt needed env vars based on config."""
    needed_vars = {}
    for var in [v[2:-1] for v in config.get('script', {}).get('env', {}).values()
                if isinstance(v, str) and var.startswith('${') and var.endswith('}')] :
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Missing required env var: {var}")
        needed_vars[var] = value

    key = Fernet.generate_key()
    return Fernet(key).encrypt(json.dumps(needed_vars).encode()), key

def validate_model_with_api(model_name: str, api_token: str, logger: logging.Logger) -> bool:
    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://huggingface.co/api/models/{model_name}"

    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            logger.info(f"Model '{model_name}' exists on the Hugging Face Hub!")
            return True
        elif response.status_code == 404:
            logger.error(f"Model '{model_name}' does not exist on the Hugging Face Hub.")
            return False
        else:
            logger.warning(f"Unexpected response while validating model '{model_name}': {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while validating model '{model_name}': {e}", exc_info=True)
        return False

def validate_dataset_with_api(dataset_name: str, api_token: str, logger: logging.Logger) -> bool:
    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if all(data.values()):
            logger.info(f"Dataset '{dataset_name}' is valid and fully supported!")
            return True
        else:
            logger.warning(f"Dataset '{dataset_name}' is valid but may have limited support:")
            logger.warning(json.dumps(data, indent=2))
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while validating dataset '{dataset_name}': {e}", exc_info=True)
        return False

def generate_training_script(project_dir: str, model: str, dataset: str, config: dict, logger: logging.Logger):
    script_path = os.path.join(project_dir, "script.py")
    script_content = f'''# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch",
#     "transformers",
#     "datasets>=2.14.6",
#     "accelerate",
#     "cryptography"
# ]
# ///

import json
import os
from cryptography.fernet import Fernet

def setup_env():
    with open('.env.encrypted', 'rb') as f:
        encrypted = f.read()
    with open('.env.key', 'rb') as f:
        key = f.read()

    cipher = Fernet(key)
    env_vars = json.loads(cipher.decrypt(encrypted))
    for k,v in env_vars.items():
        os.environ[k] = v

setup_env()

os.environ["HF_HOME"] = "/tmp/huggingface"
cache_dir = "/root/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load model and tokenizer
token = os.getenv('HF_API_KEY')
model = AutoModelForCausalLM.from_pretrained("{model}", token=token)
tokenizer = AutoTokenizer.from_pretrained("{model}", token=token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("{dataset}", cache_dir=cache_dir)

def tokenize(examples):
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length={config['training']['max_length']},
        return_tensors='pt'
    )
    outputs['labels'] = outputs['input_ids'].clone()
    return outputs

tokenized_dataset = dataset['train'].map(
    tokenize,
    batched=True,
    remove_columns=dataset['train'].column_names
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="{config['output_dir']}",
    overwrite_output_dir=True,
    num_train_epochs={config['training']['epochs']},
    per_device_train_batch_size={config['training']['batch_size']},
    learning_rate={config['training']['learning_rate']},
    save_steps={config['training']['save_steps']},
    save_total_limit={config['training']['save_limit']},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("{config['output_dir']}")
'''
    with open(script_path, 'w') as f:
        f.write(script_content)
    logger.info(f"Generated training script at {script_path}")

def load_hf_config(config_path="config/hf_train.yaml", logger: logging.Logger = None) -> Optional[dict]:
    if not os.path.exists(config_path):
        if logger:
            logger.error(f"HF training config not found at {config_path}")
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if logger:
        logger.debug(f"Loaded HF training config from {config_path}")
    return config

def update_project_config(project_dir: str, hf_config: dict, logger: logging.Logger):
    config_path = os.path.join(project_dir, "config.yaml")
    project_name = os.path.basename(project_dir)
    config = {
        "project_name": project_name,
        "provider": hf_config["provider"],
        "gpu": hf_config["gpu"],
        "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "script": {
            "path": "./run_script.sh",
            "env": {
                "HF_API_KEY": "${HF_API_KEY}",  # Add this
                "HF_HOME": "/tmp/huggingface",
                "CACHE_DIR": "/root/hf_cache"
            },
        },
        "ssh": {
            "key_path": "~/.ssh/id_ed25519",
        },
        "upload": {
            "local_dir": ".",
            "remote_dir": f"/root/workspace/{project_name}"
        },
        "secrets": {
            "HF_API_KEY": {
                "source": "env",
                "target": "HF_API_KEY"
            }
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Updated project config.yaml at {config_path}")

def main():
    # Define log file path for the main training script
    log_file_path = "train.log"  # Can be adjusted as needed
    setup_logging(log_file_path)  # Initialize logging
    logger = get_logger('my_logger')  # Retrieve the logger

    parser = argparse.ArgumentParser(description="Train HuggingFace models on RunPod")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--config", help="Custom training config path")
    args = parser.parse_args()

    # Validate model and dataset first
    hf_api_token = os.getenv("HF_API_KEY", "")

    logger.info(f"Validating model '{args.model}' and dataset '{args.dataset}'...")
    model_valid = validate_model_with_api(args.model, hf_api_token, logger)
    dataset_valid = validate_dataset_with_api(args.dataset, hf_api_token, logger)

    model_name = args.model.split('/')[-1]
    dataset_name = args.dataset.split('/')[-1]
    project_name = f"{model_name}_{dataset_name}"
    project_dir = os.path.join("projects", project_name)

    if not (model_valid and dataset_valid):
        logger.error("Validation failed. Aborting workflow.")
        if os.path.exists(project_dir):
            import shutil
            shutil.rmtree(project_dir)
            logger.warning(f"Cleaned up project directory: {project_dir}")
        return

    config = load_hf_config(args.config, logger) if args.config else load_hf_config(logger=logger)
    if not config:
        logger.error("Failed to load HF training config. Aborting workflow.")
        return

    initialize_project(project_name, logger)
    generate_training_script(project_dir, args.model, args.dataset, config, logger)
    update_project_config(project_dir, config, logger)
    automate_workflow(os.path.join(project_dir, "config.yaml"), logger)

if __name__ == "__main__":
    main()
