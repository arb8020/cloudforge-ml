#!/bin/bash
# run_script.sh
# Default run script for example_hf
#
# Navigate to uploaded dir
cd /root/workspace/example_hf
#
# Install uv
pip install uv
# Run some script
uv run script.py --model openai-community/gpt2 --dataset karpathy/tiny_shakespeare
echo "running script for example_hf"
