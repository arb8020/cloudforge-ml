#!/bin/bash
# run_script.sh

# Navigate to the uploaded directory
cd /root/workspace/example_gpt2

# Install required Python packages
pip install uv

# Run the GPT-2 script
script -qec "uv run gpt2_script.py" /dev/null
# uv run gpt2_script.py
