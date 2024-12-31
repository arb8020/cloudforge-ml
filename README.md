# cloudforge-ml
goal: one-click deployment for hugging face models/datasets on arbitrary cloud compute platforms

[demo.gif]

## current status
- supported providers: runpod
- low-friction workflows
  - initialize projects with one command using initialize_project.py
  - easy setup with default config.yaml and bash script that runs on your pod
  - deploy files via scp and run your scripts with one command using deploy_runpod.py
- example recipes
  - huggingface: gpt2 training with tiny-shakespeare
  - custom models: deploy models that inherit from hf architectures
  - custom projects: define your own training scripts with minimal setup

## quick start
### standard workflow
```bash
# train any HF model on any dataset with automatic cost tracking
uv run hf_train.py --model openai-community/gpt2 --dataset karpathy/tiny-shakespeare

# or use your own model/dataset
uv run initialize_hf_project.py my_project
uv run hf_train.py --model ./projects/my_project/model.py --dataset ./projects/my_project/dataset.py
```

### custom workflow
```bash
# initialize project
uv run initialize_project.py my_project

# update config.yaml, run_script.sh, script.py, files to scp over as needed
# check out example_cifar to see how you might clone and run a personal repository
uv run deploy_runpod.py --project my_project
```

### runpod setup
1. get api key: settings > api keys
2. create .env: `RUNPOD_API_KEY=your_key_here`
3. make ssh key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. add key: settings > ssh keys

## features
- automatic cost tracking and budget controls
- graceful error handling and cleanup
- environment and SSH key management
- modular design for provider expansion

## roadmap
- [5/6] core features
  - [x] runpod integration
  - [x] project initialization
  - [x] file deployment
  - [x] auto-ssh after script execution
  - [x] one-command training for HF models/datasets
  - [ ] cleaner abstractions

- [0/4] research features
  - [ ] advanced training (FSDP, checkpointing)
  - [ ] spot instances + interruption handling
  - [ ] wandb integration
  - [ ] multi-GPU support

- [0/4] infrastructure
  - [ ] provider abstraction layer
  - [ ] vast.ai support
  - [ ] aws/gcp integration
  - [ ] cost optimization

- [ ] UI/UX
  - [ ] smoother setup wizard
  - [ ] tqdm interaction with ssh stdout is a little weird
  - [ ] smarter dependency management (selectively loading transformers optional dependencies like sentencepiece )
  - [ ] notebook recipe
  - [ ] high performance recipe
  - [ ] model chat interface recipe
  - [ ] comfyui image gen recipe
