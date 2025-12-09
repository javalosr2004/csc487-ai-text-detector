import argparse
import yaml
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from dataloaders.mlm import MLMDataset, load_bookcorpus
from models import make_encoder
from tokenizer import make_tokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).unsqueeze(1)
    return mask


class MLMHead(nn.Module):
    """Projects encoder output to vocab size for token prediction."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        return self.decoder(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    book_corpus = load_bookcorpus("train")
    
    tokenizer = make_tokenizer(cfg)
    tokenizer.build_vocab(book_corpus)
    
    mlm_dataset = MLMDataset(
        book_corpus,
        tokenizer=tokenizer,
        max_len=cfg["training"]["max_seq_len"],
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id
    )
    
    train_loader = DataLoader(
        mlm_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    encoder = make_encoder(
        vocab=tokenizer.vocab_size,
        N=cfg["model"]["N"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        h=cfg["model"]["h"],
        dropout=cfg["model"]["dropout"]
    )
    encoder = encoder.to(device)

    mlm_head = MLMHead(cfg["model"]["d_model"], tokenizer.vocab_size).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(mlm_head.parameters()),
        lr=cfg["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    num_epochs = cfg["training"]["epochs"]
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        encoder.train()
        mlm_head.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask_positions = batch["mask_positions"].to(device)

            # Set labels to -100 where not masked (ignore in loss)
            mlm_labels = labels.clone()
            mlm_labels[~mask_positions] = -100

            mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)

            optimizer.zero_grad()

            # Get encoder output
            encoder_output = encoder(input_ids, mask)

            # Predict tokens
            logits = mlm_head(encoder_output)

            # Compute loss only on masked tokens
            loss = criterion(logits.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

        # Save encoder checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"pretrained_encoder_epoch_{epoch+1}.pt")
        torch.save(encoder.state_dict(), checkpoint_path)
        print(f"Saved encoder checkpoint to {checkpoint_path}")