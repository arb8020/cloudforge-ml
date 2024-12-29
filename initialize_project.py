# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "rich",
# ]
# ///

import os
import yaml
import argparse
from rich import print as rprint

def initialize_project(project_name: str):
    # Define project directory and default files
    project_dir = os.path.join("projects", project_name)
    config_path = os.path.join(project_dir, "config.yaml")
    script_path = os.path.join(project_dir, "run_script.sh")
    python_script_path = os.path.join(project_dir, "script.py")

    # Create the project directory
    os.makedirs(project_dir, exist_ok=True)

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
        rprint(f"[green]Created default config.yaml at {config_path}[/green]")

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
        rprint(f"[green]Created default run_script.sh at {script_path}[/green]")

    # Create a default script.py
    hello_world_script = """# script.py
# Default Python script for {project_name}

def main():
    print("Hello, World! This is the default script for {project_name}.")

if __name__ == "__main__":
    main()
"""
    if not os.path.exists(python_script_path):
        with open(python_script_path, "w") as python_script_file:
            python_script_file.write(hello_world_script.format(project_name=project_name))
        rprint(f"[green]Created default script.py at {python_script_path}[/green]")

    rprint(f"[blue]Project {project_name} initialized successfully in {project_dir}.[/blue]")

def main():
    parser = argparse.ArgumentParser(description="Initialize a new RunPod project.")
    parser.add_argument("project_name", help="The name of the project to initialize.")
    args = parser.parse_args()

    initialize_project(args.project_name)

if __name__ == "__main__":
    main()
