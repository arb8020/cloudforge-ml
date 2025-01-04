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
# run_comfy.py

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
import json
from cryptography.fernet import Fernet
import shutil
from typing import Optional

# Load environment variables from .env file
load_dotenv()

def encrypt_env(config: dict) -> tuple[bytes, bytes]:
    """Encrypt needed env vars based on config."""
    needed_vars = {}
    for var in [v[2:-1] for v in config.get('script', {}).get('env', {}).values()
                if isinstance(v, str) and v.startswith('${') and v.endswith('}')]:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Missing required env var: {var}")
        needed_vars[var] = value

    key = Fernet.generate_key()
    return Fernet(key).encrypt(json.dumps(needed_vars).encode()), key

def load_comfy_config(config_path="config/comfy.yaml", logger: logging.Logger = None) -> Optional[dict]:
    if not os.path.exists(config_path):
        if logger:
            logger.error(f"Comfy config not found at {config_path}")
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if logger:
        logger.debug(f"Loaded Comfy config from {config_path}")
    return config

def update_project_config(project_dir: str, comfy_config: dict, logger: logging.Logger):
    config_path = os.path.join(project_dir, "config.yaml")
    project_name = comfy_config.get("project_name", "comfy_project")
    config = {
        "project_name": project_name,
        "provider": comfy_config.get("provider", "runpod"),
        "gpu": comfy_config.get("gpu", "NVIDIA GeForce RTX 3090"),
        "budget": {
            "max_dollars": comfy_config.get('budget', {}).get('max_dollars', 10),
            "max_hours": comfy_config.get('budget', {}).get('max_hours', 4),
        },
        "image": comfy_config.get("image", "aitrepreneur/comfyui:latest"),
        "template_id": comfy_config.get("template_id", ""),
        "containerDiskInGb": comfy_config.get("containerDiskInGb", 10),
        "gpuCount": comfy_config.get("gpuCount", 1),
        "minMemoryInGb": comfy_config.get("minMemoryInGb", 125),
        "startJupyter": comfy_config.get("startJupyter", True),
        "startSsh": comfy_config.get("startSsh", True),
        "volumeInGb": comfy_config.get("volumeInGb", 80),
        "minVcpuCount": comfy_config.get("minVcpuCount", 16),
        "ports": comfy_config.get("ports", "3000/http,8000/http,8888/http,2999/http,7777/http,22/tcp"),
        "script": {
            "path": comfy_config.get("script", {}).get("path", "./run_script.sh"),
            "env": {
                "CACHE_DIR": comfy_config.get("script", {}).get("env", {}).get("CACHE_DIR", "/workspace/test_comfy/hf_cache"),
                "HF_HOME": comfy_config.get("script", {}).get("env", {}).get("HF_HOME", "/tmp/huggingface")
            },
        },
        "ssh": {
            "key_path": comfy_config.get("ssh", {}).get("key_path", "~/.ssh/id_ed25519"),
        },
        "upload": {
            "local_dir": comfy_config.get("upload", {}).get("local_dir", "."),
            "remote_dir": comfy_config.get("upload", {}).get("remote_dir", f"/root/workspace/{project_name}")
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Updated project config.yaml at {config_path}")

def main():
    # Define log file path for the main run_comfy script
    log_file_path = "run_comfy.log"  # Can be adjusted as needed
    setup_logging(log_file_path)  # Initialize logging
    logger = get_logger('run_comfy_logger')  # Retrieve the logger

    parser = argparse.ArgumentParser(description="Run ComfyUI on RunPod")
    parser.add_argument(
        "--config",
        default="config/comfy.yaml",
        help="Path to the ComfyUI config file (default: config/comfy.yaml)"
    )
    # --keep-alive is default, so no need to add as an argument
    args = parser.parse_args()

    # Load ComfyUI configuration
    comfy_config = load_comfy_config(args.config, logger)
    if not comfy_config:
        logger.error("Failed to load ComfyUI config. Aborting workflow.")
        sys.exit(1)

    project_name = comfy_config.get("project_name", "comfy_project")
    project_dir = os.path.join("projects", project_name)  # Use string path

    initialize_project(project_name, logger)

    # Encrypt environment variables and save keys
    try:
        encrypted_env, key = encrypt_env(comfy_config)
        with open(os.path.join(project_dir, '.env.encrypted'), 'wb') as f:
            f.write(encrypted_env)
        with open(os.path.join(project_dir, '.env.key'), 'wb') as f:
            f.write(key)
        logger.info("Encrypted environment variables and saved keys.")
    except Exception as e:
        logger.error(f"Failed to encrypt environment variables: {e}", exc_info=True)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            logger.warning(f"Cleaned up project directory due to failure: {project_dir}")
        sys.exit(1)

    update_project_config(project_dir, comfy_config, logger)

    # Path to the updated config.yaml
    config_yaml_path = os.path.join(project_dir, "config.yaml")

    # Initialize a state dictionary to track pod_id and keep_alive
    state = {
        'pod_id': None,
        'keep_alive': True  # Default to keep_alive
    }

    # Register signal handlers to ensure graceful shutdown
    def signal_handler(signum, frame):
        handle_signal(signum, frame, logger, state)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Automate workflow with keep_alive set to True by default
    automate_workflow(config_yaml_path, logger, state['keep_alive'], state)

if __name__ == "__main__":
    main()
