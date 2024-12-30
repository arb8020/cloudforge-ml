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
import torch
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   Trainer,
   TrainingArguments,
   DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Environment setup
os.environ["HF_HOME"] = "/tmp/huggingface"
cache_dir = "/root/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

def train_model(
   model_name="gpt2",
   dataset_name="tiny_shakespeare",
   output_dir="./trained_model",
   batch_size=32,
   num_epochs=3
):
   # Load model and tokenizer
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   # Handle tokenizer padding
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

   # Load dataset
   dataset = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)

   # Tokenize dataset with labels for causal LM
   def tokenize(examples):
       # Tokenize the text
       outputs = tokenizer(
           examples['text'],
           truncation=True,
           padding='max_length',
           max_length=128,
           return_tensors='pt'  # Return PyTorch tensors
       )

       # Set labels same as input_ids for causal LM
       outputs['labels'] = outputs['input_ids'].clone()

       return outputs

   tokenized_dataset = dataset['train'].map(
       tokenize,
       batched=True,
       remove_columns=dataset['train'].column_names
   )

   # Data collator
   data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer,
       mlm=False  # Not using masked language modeling
   )

   # Training arguments
   training_args = TrainingArguments(
       output_dir=output_dir,
       overwrite_output_dir=True,
       num_train_epochs=num_epochs,
       per_device_train_batch_size=batch_size,
       save_steps=10_000,
       save_total_limit=2,
   )

   # Initialize trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_dataset,
       data_collator=data_collator
   )

   # Train
   trainer.train()

   # Save final model
   trainer.save_model()
   tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--model", default="gpt2", help="HF model name")
   parser.add_argument("--dataset", default="tiny_shakespeare", help="HF dataset name")
   args = parser.parse_args()

   train_model(
       model_name=args.model,
       dataset_name=args.dataset
   )
