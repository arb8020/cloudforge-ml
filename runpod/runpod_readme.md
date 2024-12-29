# Get API key from RunPod dashboard: Settings > API Keys

# Create .env with API key

echo "RUNPOD_API_KEY=your_key_here" > .env

# Create SSH key using the email you used on runpod

ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key in RunPod dashboard: Settings > SSH Keys

# Example Workflow
1. create new pod
2. manage pod
3. deploy file
4. manage pod
5. interactive ssh session
6. pip install uv
7. uv run gpt2_script.py
