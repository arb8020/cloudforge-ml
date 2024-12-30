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
- [x] custom projects: define your own projects and scripts to run flexible experiments with a little more setup overhead
- [x] standard hf models/datasets: one-command deployment for training existing hf models on standard datasets
- [ ] custom models/datasets: deploy models that inherit from hf architectures, train using data formatted to hf dataset specs
- [ ] TBD

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
4. ssh back into your instance to continue working (see .pod_ssh)

### example huggingface workflow
1. uv run hf_train.py --model= --dataset=
4. ssh back into your instance to continue working (see .pod_ssh)

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
- [ ] captures + send back logs
- [ ] --keep-alive arg (default is to tear down on failed deploy or after sending logs)
- [ ] configurable memory
- [ ] clean up file structures/abstractions

[0/3] huggingface integration
- [ ] custom hf model template
- [ ] custom hf dataset template
- [ ] one command training
- [ ] hyperparam overrides

[0/r] devops stuff
- [ ] reading outputs while script running still delayed/clunky
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
