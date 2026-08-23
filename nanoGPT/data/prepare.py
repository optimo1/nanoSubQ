import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Config
DATASET_NAME = "HuggingFaceTB/smoltalk"
SUBSET = "all"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading/Loading SmolTalk 'all' dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME, SUBSET, split="train")

# Base GPT-2 encoding
base_enc = tiktoken.get_encoding("gpt2")

# Define ChatML special tokens
special_tokens = {
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
}

# Encoder that accepts the special ChatML tokens
enc = tiktoken.Encoding(
    name="gpt2_chatml",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **special_tokens}
)

def process_example(example):
    """Formats each turn into ChatML and tokenizes it."""
    ids = []
    for msg in example['messages']:
        role = msg['role']
        content = msg['content']
        formatted = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        ids.extend(enc.encode(formatted, allowed_special="all"))
    return ids

print("Tokenizing dataset into ChatML format...")
all_tokens = []
for example in tqdm(dataset):
    all_tokens.extend(process_example(example))

all_tokens = np.array(all_tokens, dtype=np.uint32)
print(f"Total tokens generated: {len(all_tokens):,}")

# 95% Train / 5% Validation split
split_idx = int(len(all_tokens) * 0.95)
train_ids = all_tokens[:split_idx]
val_ids = all_tokens[split_idx:]

train_ids.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
val_ids.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

print(f"\nDone! Datasets exported to:")
print(f" - {OUTPUT_DIR}/train.bin ({len(train_ids):,} tokens)")
print(f" - {OUTPUT_DIR}/val.bin ({len(val_ids):,} tokens)")