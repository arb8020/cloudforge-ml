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
- custom projects: define your own projects and scripts to run flexible experiments with a little setup overhead
- standard hf models/datasets: one-command deployment for training existing hf models on standard datasets
- custom models/datasets: deploy models that inherit from hf architectures, train using data formatted to hf dataset specs

## usage
### runpod setup
1. get api key: settings > api keys
2. create .env: `RUNPOD_API_KEY=your_key_here`
3. make ssh key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. add key: settings > ssh keys

### example custom workflow
1. uv run initialize_project.py  project_name
2. update the config.yaml, run_script.sh, script.py as desired
3. uv run deploy_runpod.py --project=project_name
4. use --keep-alive to ssh into the instance after running the script

### huggingface setup
5. if planning to use gated models, add your huggingface token to HF_API_KEY=your_key_here

### example huggingface workflow
1. uv run initialize_hf_project.py project_name
2. adjust your model and dataset code
3. uv run hf_train.py --model=./projects/project_name/my_hf_model.py --dataset=./projects/project_name/my_hf_dataset.py
4. use --keep-alive to ssh into the instance after running the script

### example custom huggingface workflow
1. uv run hf_train.py --model=openai-community/gpt2 --dataset=karpathy/tiny-shakespeare (use hf defined names)
2. use --keep-alive to ssh into the instance after running the script


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
- [x] auto-ssh after script execution
- [x] one-command train existing hf model on existing hf dataset recipe (hf_train)
- [x] git clone project and run code recipe (projects/example_cifar)
- [x] captures + send back logs
- [x] --keep-alive arg (default is to tear down on failed deploy or after sending logs)
- [ ] easier to kill a keep alive pod
- [ ] clean up file structures/abstractions
- [ ] ruff/astral

[0/3] huggingface integration
- [ ] example custom hf model template
- [ ] examle custom hf dataset template
- [ ] one command training of custom hf model + dataset
- [ ] training hyperparam overrides

[0/5] devops stuff
- [ ] testing
- [ ] reading outputs while script running still delayed/clunky
- [ ] smarter dependency management (when to load/not load sentencepiece, etc)
- [ ] wandb/etc integration (?)
- [ ] spot instance + checkpointing support (use a spot instance, checkpoint and provision new spot if interrupt)
- [ ] bug: runpod secrets doesn't work with exposed tcp port (had to encrypt + decrypt file)

[0/6] cost/etc
- [ ] budget parameters
- [ ] cost estimation
- [ ] vast.ai integration
- [ ] provider abstraction layer
- [ ] cost optimization
- [ ] aws/gcp/???

[0/4] UX
- [ ] more interactive project setup wizard
- [ ] notebook support
- [ ] high perf recipe (multi-gpu, fsdp + compile)
- [ ] model chat recipe
- [ ] comfyui recipe

- [0/2] more recipes
  - [ ] model chat interface
  - [ ] comfyui image gen
