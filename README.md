# cloudforge-ml
goal: ~one-click deployment for hugging face models/datasets on arbitrary cloud compute platforms

[demo.gif] # tbd

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

### runpod/ssh setup (required)
1. get api key: settings > api keys (https://www.runpod.io/console/user/settings)
2. create .env: `RUNPOD_API_KEY=your_key_here`
3. make ssh key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. add key: settings > ssh keys (https://www.runpod.io/console/user/settings)

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

## features
- automatic cost tracking and budget controls
- graceful error handling and cleanup
- environment and SSH key management
- modular design for provider expansion

## detailed usage info
### deploying a custom project to RunPod
if you want to create a custom project, run:

```bash
uv run initialize_project.py my_project
```

this will create projects/my_project with:
- config.yaml (your deployment config)
- run_script.sh (default bash script to run on the pod)
- script.py (a sample python script)
from there, edit script.py, add files, install dependencies, etc.

once you're ready you can use
'''bash
uv run deploy_runpod.py --project=my_project
'''
this will
- look up projects/my_project/config.yaml
- create a new pod on RunPod
- upload the files in your project directory
- execute run_script.sh
- terminate the pod on completion/failed deployment unless --keep-alive is specified

### training a huggingface model
if you want to train a text generation model with huggingface

```bash
uv run hf_train.py --model <HF_MODEL_OR_LOCAL.py> --dataset <HF_DATASET_OR_LOCAL.py> [--keep-alive]
```

example: train GPT2 on tiny_shakespeare
uv run hf_train.py --model gpt2 --dataset karpathy/tiny_shakespeare

key things to note:
- if you pass a local .py file for --model or --dataset, the script automatically copies them into your project and uses them.
- if you omit --keep-alive, the pod terminates after training. Otherwise, it’ll drop you into an SSH session when done.

### initializing a custom huggingface run
if you want to initialize training a custom text generation model with huggingface, and your own custom dataset

```bash
uv run initialize_hf_project.py my_hf
```

example: train mistral on tiny_shakespeare
uv run hf_train.py --model mistralai/Mistral-7B-Instruct-v0.3 --dataset karpathy/tiny_shakespeare

key things to note:
- if you're using a gated model (like in the example), be sure you have access and you put your huggingface token in your .env
- if you pass a local .py file for --model or --dataset, the script automatically copies them into your project and uses them.
- if you omit --keep-alive, the pod terminates after training. Otherwise, it’ll drop you into an SSH session when done.

### examples to help you get started
There are a few sample projects in projects/:

example_cifar: clones a CIFAR10 speedrun repository, installs deps, and runs training.

```bash
uv run deploy_runpod.py --project=example_cifar
```

example_gpt2: sets up training for gpt2 on the tiny shakespeare dataset

```bash
uv run deploy_runpod.py --project=example_gpt2
```

example_gpt2: basic GPT-2 text training script with HF + datasets.
example_hf: Another example that demonstrates using hf_train.py with a local script.

## roadmap
- [5/5] core features
  - [x] runpod integration
  - [x] project initialization
  - [x] file deployment
  - [x] auto-ssh after script execution
  - [x] one-command training for HF models/datasets
  - [x] one-command initialization for ComfyUI text2img workflows

- [0/4] UI/UX
  - [ ] better abstractions/code organization for continuing work (extracting templates, etc)
  - [ ] better pod naming
  - [ ] smarter dependency management (selectively loading transformers optional dependencies like sentencepiece )
  - [ ] smoother setup wizard
  - [ ] tqdm interaction with ssh stdout is a little weird

- [0/5] research features
  - [ ] support for tasks beyond text generation
  - [ ] advanced training (FSDP, checkpointing)
  - [ ] spot instances + interruption handling
  - [ ] wandb integration
  - [ ] multi-GPU support

- [0/4] infrastructure
  - [ ] provider abstraction layer
  - [ ] vast.ai support
  - [ ] aws/gcp integration
  - [ ] cost optimization

- [ ] huggingface
  - [ ] support for tasks other than text generation
  - [ ] pre + post training pipeline

- [ ] comfyui
  - [ ] customizable dockerfile/runpod template to change bootup behavior (automatically downloaded models, etc)
  - [ ] bootup with more example workflows

- [0/4] recipes
  - [ ] notebook recipe
  - [ ] high performance recipe
  - [ ] model chat interface recipe
