'''
This module serves for the masked language modeling training step.
'''

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
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


def load_bookcorpus(split, max_samples):
    """Load BookCorpus dataset from Hugging Face."""
    print("Loading BookCorpus dataset...")
    full_dataset = None
    try:
        full_dataset = load_dataset("bookcorpus", split=split)
    except Exception as e:
        print(f"Error loading bookcorpus: {e}")
        print("Trying alternative dataset...")
        # Alternative: use a smaller subset or different dataset
        full_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    subset_dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
    
    texts = [item["text"] for item in subset_dataset if item.get("text") and len(item["text"]) > 50]
    print(f"Loaded {len(texts)} texts from BookCorpus")
    
    return texts


def load_bookcorpus_train_val(max_samples, val_ratio=0.02, seed=42, min_length=50):
    """
    Load BookCorpus (or fallback dataset) and deterministically split into train/val.
    """

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1); got {val_ratio}")

    texts = load_bookcorpus(split="train", max_samples=max_samples)
    texts = [t for t in texts if t and len(t) > min_length]

    if len(texts) < 2:
        raise ValueError("Not enough samples to create a train/val split.")

    rng = random.Random(seed)
    indices = range(len(texts))
    rng.shuffle(indices)

    val_size = max(1, int(round(len(indices) * val_ratio)))
    val_idx = (indices[:val_size])

    train_texts = [t for i, t in enumerate(texts) if i not in val_idx]
    val_texts = [t for i, t in enumerate(texts) if i in val_idx]

    return train_texts, val_texts