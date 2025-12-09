'''
This module serves for the masked language modeling training step.
'''

import torch
from datasets import load_dataset, Dataset
import random

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling pre-training."""
    def __init__(self, texts: list[str], tokenizer, max_len, mask_prob=0.15, mask_token_id=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id if mask_token_id else tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], self.max_len)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create labels (only compute loss on masked tokens)
        labels = input_ids.clone()
        
        # Create mask: 80% [MASK], 10% random token, 10% unchanged
        mask_positions = torch.rand(len(tokens)) < self.mask_prob
        # Don't mask [CLS] or [PAD] tokens
        mask_positions[0] = False  # Don't mask [CLS]
        mask_positions[input_ids == self.tokenizer.pad_token_id] = False
        
        # Apply masking strategy
        for i, should_mask in enumerate(mask_positions):
            if should_mask:
                rand = random.random()
                if rand < 0.8:
                    # 80%: replace with [MASK]
                    input_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    # 10%: replace with random token
                    input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10%: keep original (no change)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "mask_positions": mask_positions
        }


def load_bookcorpus(split="train", max_samples=None):
    """Load BookCorpus dataset from Hugging Face."""
    print("Loading BookCorpus dataset...")
    try:
        dataset = load_dataset("bookcorpus", split=split)
    except Exception as e:
        print(f"Error loading bookcorpus: {e}")
        print("Trying alternative dataset...")
        # Alternative: use a smaller subset or different dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = [item["text"] for item in dataset if item.get("text") and len(item["text"]) > 50]
    print(f"Loaded {len(texts)} texts from BookCorpus")
    
    return texts