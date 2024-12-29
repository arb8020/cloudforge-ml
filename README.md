# cloudforge-ml
goal: one-click training for hugging face models/datasets on arbitrary cloud compute platforms

current status:
- providers: runpod
- cli features:
 - create/stop/terminate pods
 - ssh tunneling
 - deploy files via scp
- see `runpod/readme.md` for setup

coming soon:
- one-click automation:
 - auto-deploy training scripts
 - simplified workflow
- model/dataset features:
 - try different model architectures 
 - experiment with datasets
 - track experiments
- more providers: 
 - aws
 - gcp
 - vast.ai
 - ???
