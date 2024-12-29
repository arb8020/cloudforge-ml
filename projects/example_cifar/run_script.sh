#!/bin/bash
# run_script.sh
# Default run script for example_cifar
#
# Navigate to uploaded dir
cd /root/workspace/example_cifar
#
REPO_URL="https://github.com/tysam-code/hlb-CIFAR10.git"
BRANCH="main"

# Extract repo name from URL
REPO_NAME=$(basename "$REPO_URL" .git)

# Clone the repository into a subdirectory
echo "Cloning repository $REPO_URL (branch: $BRANCH) into $REPO_NAME..."
git clone --branch "$BRANCH" "$REPO_URL" "$REPO_NAME" || { echo "Failed to clone repo"; exit 1; }

# Navigate to the cloned repo
cd "$REPO_NAME" || { echo "Failed to navigate to $REPO_NAME"; exit 1; }
python -m pip install -r requirements.txt
python main.py
