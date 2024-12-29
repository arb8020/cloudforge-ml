# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch",
#     "transformers",
#     "datasets>=2.14.6",
#     "accelerate",
# ]
# ///

import os
os.environ["HF_HOME"] = "/tmp/huggingface"

# Create and ensure cache directory exists
cache_dir = "/root/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("tiny_shakespeare", cache_dir=cache_dir)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize dataset
def tokenize(examples):
   return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset['train'].map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
   output_dir="./shakespeare_gpt2",
   overwrite_output_dir=True,
   num_train_epochs=3,
   per_device_train_batch_size=32,
   save_steps=10_000,
   save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
   tokenizer=tokenizer,
   mlm=False
)

# Initialize trainer
trainer = Trainer(
   model=model,
   args=training_args,
   data_collator=data_collator,
   train_dataset=tokenized_dataset
)

# Train
trainer.train()
