import os
import time
import re
import json
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, DatasetDict

# Set environment variables for Hugging Face
os.environ["HF_HOME"] = "/media/data_2/datasets/large_models_data/hf_cache"

# Login to Hugging Face Hub
login()

# Download tokenizer and weights
LLAMA_SIZE_STR = '1B'
local_dir = "/media/data_2/datasets/large_models_data/llama_weights/llama32_1b"
tokenizer_file_path = hf_hub_download(
    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    filename="original/tokenizer.model",
    local_dir=local_dir
)
# weights_file = hf_hub_download(
#     repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
#     filename=f"model.safetensors",
#     local_dir=local_dir
# )

# Load dataset and split into train, validation, and test sets
ds = load_dataset("sedthh/gutenberg_english", cache_dir="/media/data_2/datasets/large_models_data/gutenberg_data")
train_test_split = ds["train"].train_test_split(test_size=0.3, seed=442)
val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=442)

splits = DatasetDict({
    "train": train_test_split["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"]
})

# Save splits to parquet
splits["train"].to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/train.parquet")
splits["validation"].to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/val.parquet")
splits["test"].to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/test.parquet")

# Initialize tokenizer
from llama3_utils import Llama32Tokenizer

tokenizer = Llama32Tokenizer(tokenizer_model_path=tokenizer_file_path)
separator = tokenizer.decode([tokenizer.tik_model.eot_token])

# Load parquet files
train_pd = pd.read_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/train.parquet")
val_pd = pd.read_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/val.parquet")
test_pd = pd.read_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/test.parquet")

# Preprocess and tokenize datasets
print("Preprocessing and tokenizing data...")
def process_and_tokenize(data_set, name):
    data_set['TEXT'] = data_set['TEXT'].apply(lambda x: x.strip())
    text = separator.join(data_set['TEXT'])
    chunk_size = 100000
    tokens = []
    for i in tqdm(range(0, len(text), chunk_size), desc=f"Tokenizing {name}"):
        chunk = text[i:i+chunk_size]
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        chunk = re.sub(r"\n{2,}", "\n\n", chunk)
        tokens.extend(tokenizer.encode(chunk, allowed_special={separator}))
    return tokens

train_tokens = process_and_tokenize(train_pd, "train")
val_tokens = process_and_tokenize(val_pd, "val")
test_tokens = process_and_tokenize(test_pd, "test")

# Save tokenized datasets to parquet
pd.DataFrame({"train": train_tokens}).to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/train_tokenized.parquet")
pd.DataFrame({"val": val_tokens}).to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/val_tokenized.parquet")
pd.DataFrame({"test": test_tokens}).to_parquet("/media/data_2/datasets/large_models_data/gutenberg_data/test_tokenized.parquet")

print("Tokenized data saved successfully.")
