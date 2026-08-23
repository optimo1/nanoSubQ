import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceTB/smoltalk"
SUBSET = "all"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Setup ChatML Tokenizer
base_enc = tiktoken.get_encoding("gpt2")
special_tokens = {
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
}
enc = tiktoken.Encoding(
    name="gpt2_chatml",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **special_tokens}
)

def process_example(example):
    ids = []
    for msg in example['messages']:
        role = msg['role']
        content = msg['content']
        formatted = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        ids.extend(enc.encode(formatted, allowed_special="all"))
    return ids

print("Downloading dataset...")
dataset = load_dataset(DATASET_NAME, SUBSET, split="train")

# 2. Pre-estimate or write directly to binary via disk buffer
train_path = os.path.join(OUTPUT_DIR, "train.bin")
val_path = os.path.join(OUTPUT_DIR, "val.bin")

# Process in small chunk files on disk to eliminate RAM overhead
temp_file = os.path.join(OUTPUT_DIR, "temp_all.bin")
total_tokens = 0

print("Tokenizing and streaming directly to disk...")
with open(temp_file, "wb") as f:
    buffer = []
    for example in tqdm(dataset):
        ids = process_example(example)
        buffer.extend(ids)
        
        # Flush to disk every 100,000 tokens to keep RAM low
        if len(buffer) >= 100_000:
            arr = np.array(buffer, dtype=np.uint32)
            f.write(arr.tobytes())
            total_tokens += len(buffer)
            buffer = []
            
    # Flush remaining
    if len(buffer) > 0:
        arr = np.array(buffer, dtype=np.uint32)
        f.write(arr.tobytes())
        total_tokens += len(buffer)
        buffer = []

print(f"Total tokens generated: {total_tokens:,}")

# 3. Split into train.bin (95%) and val.bin (5%) on disk
split_idx = int(total_tokens * 0.95)

print("Splitting into train.bin and val.bin...")
all_memmap = np.memmap(temp_file, dtype=np.uint32, mode='r')

# Copy train split
train_memmap = np.memmap(train_path, dtype=np.uint32, mode='w+', shape=(split_idx,))
train_memmap[:] = all_memmap[:split_idx]
train_memmap.flush()

# Copy val split
val_memmap = np.memmap(val_path, dtype=np.uint32, mode='w+', shape=(total_tokens - split_idx,))
val_memmap[:] = all_memmap[split_idx:]
val_memmap.flush()

# Clean up temporary file
del all_memmap, train_memmap, val_memmap
os.remove(temp_file)

print("Done! Low-RAM tokenization complete.")