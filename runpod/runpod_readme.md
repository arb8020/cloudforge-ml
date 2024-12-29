# runpod setup
## setup
1. get api key: settings > api keys
2. create .env: `RUNPOD_API_KEY=your_key_here`
3. make ssh key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. add key: settings > ssh keys

## example workflow
1. create pod (NVIDIA A40)
2. deploy files (gpt2_script.py)
3. ssh into pod
4. install uv: `pip install uv`
5. run: `uv run script.py`

## requirements
```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch",
#     "transformers",
#     "datasets",
# ]
# ///
