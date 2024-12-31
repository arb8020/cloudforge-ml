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
import signal
import sys
from logging_setup import setup_logging, get_logger
from initialize_project import initialize_project
from deploy_runpod import automate_workflow, handle_signal
import requests
from cryptography.fernet import Fernet
import json
from typing import Optional
import shutil

# Load environment variables from .env file
load_dotenv()

def encrypt_env(config: dict) -> tuple[bytes, bytes]:
    """Encrypt needed env vars based on config."""
    needed_vars = {}
    # Fixed the variable name from 'var' to 'v' in the list comprehension
    for var in [v[2:-1] for v in config.get('script', {}).get('env', {}).values()
                if isinstance(v, str) and v.startswith('${') and v.endswith('}')]:
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

def generate_training_script(
    project_dir: str,
    model: str,
    dataset: str,
    config: dict,
    logger: logging.Logger,
    is_local_model: bool = False,
    is_local_dataset: bool = False
):
    script_path = os.path.join(project_dir, "script.py")

    script_content = f'''# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch",
#     "transformers",
#     "datasets>=2.14.6",
#     "accelerate",
#     "cryptography",
# ]
# ///

import json
import os
from cryptography.fernet import Fernet
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import load_dataset

def setup_env():
    with open('.env.encrypted', 'rb') as f:
        encrypted = f.read()
    with open('.env.key', 'rb') as f:
        key = f.read()

    cipher = Fernet(key)
    env_vars = json.loads(cipher.decrypt(encrypted))
    for k, v in env_vars.items():
        os.environ[k] = v

def main():
    print("Starting training script.")

    setup_env()
    print("Environment variables loaded and decrypted.")

    os.environ["HF_HOME"] = "/tmp/huggingface"
    cache_dir = "/root/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory set at {{cache_dir}}.")

    token = os.getenv('HF_API_KEY')
    if not token:
        print("Error: HF_API_KEY not found in environment variables.")
        raise EnvironmentError("HF_API_KEY not set.")

    # Model, tokenizer and dataset setup
    if {is_local_model}:
        print(f"Loading local model from {model}.")
        import sys
        sys.path.append(os.path.dirname("{model}"))
        from model import TinyLLM, TinyLLMConfig
        config_model = TinyLLMConfig()
        model = TinyLLM(config_model)
        print("Local model loaded successfully.")

        if {is_local_dataset}:
            print(f"Loading local dataset from {dataset}.")
            sys.path.append(os.path.dirname("{dataset}"))
            from dataset import SyntheticDataset
            dataset = SyntheticDataset("gpt2")
            tokenizer = dataset.tokenizer
            tokenized_dataset = dataset
            print("Local dataset loaded successfully.")
        else:
            print(f"Loading dataset '{{dataset}}' from Hugging Face.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=token)
            dataset = load_dataset("{dataset}", cache_dir=cache_dir)
            tokenized_dataset = dataset['train'].map(
                lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
                batched=True,
                remove_columns=dataset['train'].column_names
            )
            print("Dataset loaded and tokenized successfully.")
    else:
        print(f"Loading model '{{model}}' from Hugging Face.")
        model = AutoModelForCausalLM.from_pretrained("{model}", use_auth_token=token)
        tokenizer = AutoTokenizer.from_pretrained("{model}", use_auth_token=token)
        print("Model and tokenizer loaded successfully.")

        if {is_local_dataset}:
            print(f"Loading local dataset from {dataset}.")
            sys.path.append(os.path.dirname("{dataset}"))
            from dataset import SyntheticDataset
            dataset = SyntheticDataset(tokenizer.name_or_path)
            tokenized_dataset = dataset
            print("Local dataset loaded successfully.")
        else:
            print(f"Loading dataset '{{dataset}}' from Hugging Face.")
            dataset = load_dataset("{dataset}", cache_dir=cache_dir)
            tokenized_dataset = dataset['train'].map(
                lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
                batched=True,
                remove_columns=dataset['train'].column_names
            )
            print("Dataset loaded and tokenized successfully.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token was None. Set pad_token to eos_token.")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    print("Data collator initialized.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="{config['output_dir']}",
        overwrite_output_dir=True,
        num_train_epochs={config['training']['epochs']},
        per_device_train_batch_size={config['training']['batch_size']},
        learning_rate={config['training']['learning_rate']},
        save_steps={config['training']['save_steps']},
        save_total_limit={config['training']['save_limit']},
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"  # Disable default reporting to avoid conflicts
    )
    print("Training arguments set.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    print("Trainer initialized.")
    print("Model vocab size:", model.config.vocab_size)
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    print("Starting training.")
    trainer.train()

    print("Training completed.")

    print("Saving model.")
    trainer.save_model()
    tokenizer.save_pretrained("{config['output_dir']}")
    print(f"Model and tokenizer saved to {config['output_dir']}.")

if __name__ == "__main__":
    main()
'''

    with open(script_path, 'w') as f:
        f.write(script_content)
    logger.info(f"Generated training script with print statements at {script_path}")


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
    parser.add_argument(
            "--model",
            required=True,
            help="HuggingFace model name or path to local model.py"
        )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name or path to local dataset.py"
    )
    parser.add_argument("--config", help="Custom training config path")
    # Added --keep-alive argument
    parser.add_argument(
        "--keep-alive",
        action='store_true',
        default=False,
        help="If set, keeps the pod running and drops into an SSH session after training. Otherwise, terminates the pod after training."
    )
    args = parser.parse_args()

    # Extract the keep_alive flag
    keep_alive = args.keep_alive

    # Validate model and dataset first
    is_local_model = os.path.isfile(args.model)
    is_local_dataset = os.path.isfile(args.dataset)

    hf_api_token = os.getenv("HF_API_KEY", "")
    if not is_local_model:
        logger.info(f"Validating model '{args.model}'...")
        model_valid = validate_model_with_api(args.model, hf_api_token, logger)
    else:
        model_valid = True
        logger.info(f"Using local model from {args.model}")

    if not is_local_dataset:
        logger.info(f"Validating  dataset '{args.dataset}'...")
        dataset_valid = validate_dataset_with_api(args.dataset, hf_api_token, logger)
    else:
        dataset_valid = True
        logger.info(f"Using local dataset from {args.dataset}")

    if os.path.isfile(args.model):
       model_name = Path(args.model).stem  # gets filename without extension
    else:
       model_name = args.model.split('/')[-1]

    if os.path.isfile(args.dataset):
       dataset_name = Path(args.dataset).stem
    else:
       dataset_name = args.dataset.split('/')[-1]

    project_name = f"{model_name}_{dataset_name}"
    project_dir = os.path.join("projects", project_name)

    if not (model_valid and dataset_valid):
        logger.error("Validation failed. Aborting workflow.")
        if os.path.exists(project_dir):
            import shutil
            shutil.rmtree(project_dir)
            logger.warning(f"Cleaned up project directory: {project_dir}")
        sys.exit(1)  # Changed to sys.exit for consistency

    config = load_hf_config(args.config, logger) if args.config else load_hf_config(logger=logger)
    if not config:
        logger.error("Failed to load HF training config. Aborting workflow.")
        sys.exit(1)

    initialize_project(project_name, logger)
    generate_training_script(project_dir, args.model, args.dataset, config, logger, is_local_model, is_local_dataset)
    update_project_config(project_dir, config, logger)

    # Path to the updated config.yaml
    config_yaml_path = os.path.join(project_dir, "config.yaml")

    # Initialize a state dictionary to track pod_id and keep_alive
    state = {
        'pod_id': None,
        'keep_alive': keep_alive
    }

    # Register signal handlers to ensure graceful shutdown
    def signal_handler(signum, frame):
        handle_signal(signum, frame, hf_api_token, logger, state)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Propagate the --keep-alive flag to automate_workflow
    automate_workflow(config_yaml_path, logger, keep_alive, state)

if __name__ == "__main__":
    main()
