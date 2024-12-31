# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "python-dotenv"
# ]
# ///
# custom_hf_initialize.py

import os
import yaml
import logging
from pathlib import Path
from logging_setup import setup_logging, get_logger
import argparse

def initialize_custom_hf_project(project_name: str, logger: logging.Logger):
   """Creates template for custom HF model/dataset development"""
   project_dir = os.path.join("projects", project_name)
   os.makedirs(project_dir, exist_ok=True)

   # Create model template
   model_path = os.path.join(project_dir, "my_hf_model.py")
   model_template = '''
from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
from typing import Optional, Dict

class TinyLLMConfig(PretrainedConfig):
   model_type = "tiny_llm"

   def __init__(
       self,
       hidden_size: int = 128,     # tiny for testing
       num_layers: int = 2,
       num_heads: int = 4,
       vocab_size: int = 50257,
       max_position_embeddings: int = 1024,
       **kwargs,
   ):
       super().__init__(**kwargs)
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.num_heads = num_heads
       self.vocab_size = vocab_size
       self.max_position_embeddings = max_position_embeddings
       self.head_dim = hidden_size // num_heads

class TinyLLM(PreTrainedModel):
   config_class = TinyLLMConfig

   def __init__(self, config: TinyLLMConfig):
       super().__init__(config)
       self.config = config

       self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
       self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

       # Simple single-head attention + MLP layers
       self.layers = nn.ModuleList([
           nn.TransformerEncoderLayer(
               d_model=config.hidden_size,
               nhead=config.num_heads,
               dim_feedforward=config.hidden_size * 4,
               batch_first=True
           ) for _ in range(config.num_layers)
       ])

       self.norm = nn.LayerNorm(config.hidden_size)
       self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

   def forward(
       self,
       input_ids: Optional[torch.LongTensor] = None,
       attention_mask: Optional[torch.FloatTensor] = None,
       labels: Optional[torch.LongTensor] = None,
       **kwargs,
   ) -> Dict[str, torch.Tensor]:
       # Add positional embeddings
       position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
       hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)

       # Pass through transformer layers
       for layer in self.layers:
           hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)

       hidden_states = self.norm(hidden_states)
       logits = self.lm_head(hidden_states)

       loss = None
       if labels is not None:
           loss_fct = nn.CrossEntropyLoss()
           loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

       return {"loss": loss, "logits": logits}
'''

   # Create dataset template
   dataset_path = os.path.join(project_dir, "my_hf_dataset.py")
   dataset_template = '''
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import random

class SyntheticDataset(Dataset):
   """Generates random text data for testing"""

   def __init__(self,
                tokenizer_name_or_path: str,
                max_length: int = 128,
                num_samples: int = 100):
       self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
       self.max_length = max_length

       if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = self.tokenizer.eos_token

       # Generate synthetic data
       self.data = []
       words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
       for _ in range(num_samples):
           # Generate random "sentences" from word list
           text = " ".join(random.choices(words, k=random.randint(5, 15)))
           self.data.append(text)

   def __len__(self):
       return len(self.data)

   def __getitem__(self, idx):
       text = self.data[idx]

       encodings = self.tokenizer(
           text,
           truncation=True,
           max_length=self.max_length,
           padding="max_length",
           return_tensors="pt"
       )

       return {
           "input_ids": encodings.input_ids[0],
           "attention_mask": encodings.attention_mask[0],
           "labels": encodings.input_ids[0].clone()
       }
'''

   # Create config template
   config_path = os.path.join(project_dir, "config.yaml")
   config_template = '''
provider: runpod
gpu: A40
budget:
 max_dollars: 10
 max_hours: 4
training:
 batch_size: 8  # small for testing
 epochs: 2
 learning_rate: 3e-4
 max_length: 128
 save_steps: 100
 save_limit: 2
output_dir: ./trained_model
'''

   # Write all templates if they don't exist
   if not os.path.exists(model_path):
       with open(model_path, "w") as f:
           f.write(model_template)
       logger.info(f"Created model template at {model_path}")

   if not os.path.exists(dataset_path):
       with open(dataset_path, "w") as f:
           f.write(dataset_template)
       logger.info(f"Created dataset template at {dataset_path}")

   if not os.path.exists(config_path):
       with open(config_path, "w") as f:
           f.write(config_template)
       logger.info(f"Created config template at {config_path}")

def main():
   setup_logging("custom_hf_init.log")
   logger = get_logger('custom_hf_init')

   parser = argparse.ArgumentParser(description="Initialize a custom HuggingFace project")
   parser.add_argument("project_name", help="Name of the project to create")
   args = parser.parse_args()

   initialize_custom_hf_project(args.project_name, logger)
   logger.info(f"Custom HF project {args.project_name} initialized successfully")

if __name__ == "__main__":
   main()
