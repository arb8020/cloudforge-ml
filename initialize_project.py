# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "rich",
#     "logging",
# ]
# ///
# initialize_project.py

import os
import yaml
import logging
import argparse
from pathlib import Path

# Import logging setup
from logging_setup import setup_logging, get_logger

def ensure_log_directory(log_file_path):
    log_dir = Path(log_file_path).parent
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

def initialize_project(project_name: str, logger: logging.Logger):
    # Define project directory and default files
    project_dir = os.path.join("projects", project_name)
    config_path = os.path.join(project_dir, "config.yaml")
    script_path = os.path.join(project_dir, "run_script.sh")
    python_script_path = os.path.join(project_dir, "script.py")

    # Create the project directory
    os.makedirs(project_dir, exist_ok=True)
    logger.info(f"Project directory '{project_dir}' created or already exists.")

    # Create a default config.yaml
    default_config = {
        "project_name": project_name,
        "provider": "runpod",
        "gpu": "A40",
        "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "script": {
            "path": "./run_script.sh",
            "env": {
                "HF_HOME": "/tmp/huggingface",
                "CACHE_DIR": f"/workspace/{project_name}/hf_cache"
            },
        },
        "ssh": {
            "key_path": "~/.ssh/id_ed25519",
        },
        "upload": {
            "local_dir": ".",
            "remote_dir": f"/root/workspace/{project_name}"
        },
    }
    if not os.path.exists(config_path):
        with open(config_path, "w") as config_file:
            yaml.dump(default_config, config_file)
        logger.info(f"Created default config.yaml at {config_path}")
    else:
        logger.warning(f"config.yaml already exists at {config_path}")

    # Create a default run_script.sh
    default_script = f"""#!/bin/bash
# run_script.sh
# Default run script for {project_name}
#
# Navigate to uploaded dir
cd /root/workspace/{project_name}
#
# Install uv
pip install uv
# Run some script
uv run script.py
echo "Running script for {project_name}"
"""
    if not os.path.exists(script_path):
        with open(script_path, "w") as script_file:
            script_file.write(default_script)
        os.chmod(script_path, 0o755)  # Make the script executable
        logger.info(f"Created default run_script.sh at {script_path}")
    else:
        logger.warning(f"run_script.sh already exists at {script_path}")

    # Create a default script.py
    hello_world_script = f"""# script.py
# Default Python script for {project_name}

def main():
    print("Hello, World! This is the default script for {project_name}.")

if __name__ == "__main__":
    main()
"""
    if not os.path.exists(python_script_path):
        with open(python_script_path, "w") as python_script_file:
            python_script_file.write(hello_world_script)
        logger.info(f"Created default script.py at {python_script_path}")
    else:
        logger.warning(f"script.py already exists at {python_script_path}")

    logger.info(f"Project '{project_name}' initialized successfully in {project_dir}.")

def main():
    # Define log file path for initialization logs
    log_file_path = "initialize_project.log"  # Can be adjusted as needed
    ensure_log_directory(log_file_path)
    setup_logging(log_file_path)  # Initialize logging
    logger = get_logger('my_logger')  # Retrieve the logger

    parser = argparse.ArgumentParser(description="Initialize a new RunPod project.")
    parser.add_argument("project_name", help="The name of the project to initialize.")
    args = parser.parse_args()

    initialize_project(args.project_name, logger)

if __name__ == "__main__":
    main()
