#!/bin/bash
# run_script.sh
# Default run script for my_proj
#
# Navigate to uploaded dir
cd /root/workspace/my_proj
#
# Install uv
pip install uv
# Run some script
uv run script.py
echo "Running script for my_proj"
