# cloudforge-ml
goal: one-click deployment for hugging face models/datasets on arbitrary cloud compute platforms

## current status
- supported providers: runpod
- low-friction workflows
  - initialize projects with one command using initialize_project.py
  - easy setup with default config.yaml and bash script that runs on your pod
  - deploy files via scp and run your scripts with one command using deploy_runpod.py
- example recipes
  - huggingface: gpt2 training with tiny-shakespeare

##  use cases
[x] custom projects: define your own projects and scripts to run flexible experiments with a little more setup overhead
[ ] standard hf models/datasets: one-command deployment for training existing hf models on standard datasets
[ ] custom models/datasets: deploy models that inherit from hf architectures, train using data formatted to hf dataset specs
[ ] dev mode: immediate file upload and ssh in for maximum flexibility

## usage
### runpod setup
1. get api key: settings > api keys
2. create .env: `RUNPOD_API_KEY=your_key_here`
3. make ssh key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. add key: settings > ssh keys

### example automated workflow
1. uv run initialize_project.py
2. update the config.yaml, run_script.sh, script.py as desired
3. uv run deploy_runpod.py

### example workflow in runpod_interactive
1. create pod (nvidia a40)
2. deploy files (gpt2_script.py)
3. ssh into pod
4. install uv: `pip install uv`
5. run: `uv run script.py`

# roadmap

[5/5] core features
- [x] basic runpod integration
- [x] project initialization
- [x] file deployment
- [x] development mode (auto-ssh after script execution)
- [x] train gpt2 on tiny-shakespeare recipe

[0/7] developer experience
- [0/2] basic infra
  - [ ] save outputs to file
  - [ ] better monitoring
- [0/5] more recipes (some of these might get moved down)
  - [ ] git cloning
  - [ ] multi-gpu recipes
  - [ ] cifar speedrun
  - [ ] model chat interface
  - [ ] comfyui image gen

[0/3] huggingface integration
- [ ] custom hf model template
- [ ] custom hf dataset template
- [ ] one command training

[0/4] platform/performance expansion
- [ ] vast.ai integration
- [ ] provider abstraction layer
- [ ] cost optimization features
- [ ] aws/gcp/???
- [ ] distributed training across pods (within one provider)
