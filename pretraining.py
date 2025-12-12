import argparse
import time
import yaml
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from dataloaders.mlm import MLMDataset, load_bookcorpus_train_val
from models import make_encoder
from tokenizer import make_tokenizer
from helper import load_config

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
    drive_checkpoint_dir = "/content/drive/MyDrive/pretrain_checkpoints"
    os.makedirs(drive_checkpoint_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--epochs", type=str, required=False, help="Number of epochs to train")
    parser.add_argument("--max_samples", type=str, required=False, help="Number of training samples")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to intermediate model checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint
    num_epochs = int(args.epochs) if args.epochs is not None else cfg["training"]["epochs"]
    max_samples = int(args.max_samples) if args.max_samples is not None else 10_000_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_ratio = cfg.get("training", {}).get("val_ratio", 0.02)
    train_texts, val_texts = load_bookcorpus_train_val(
        max_samples=max_samples,
        val_ratio=val_ratio,
        seed=cfg["training"]["seed"],
    )
    
    tokenizer = make_tokenizer(cfg)
    tokenizer.build_vocab(train_texts)
    
    train_dataset = MLMDataset(
        train_texts,
        tokenizer=tokenizer,
        max_len=cfg["training"]["max_seq_len"],
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id
    )

    val_dataset = MLMDataset(
        val_texts,
        tokenizer=tokenizer,
        max_len=cfg["training"]["max_seq_len"],
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    encoder = make_encoder(
        vocab=tokenizer.vocab_size,
        N=cfg["model"]["N"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        h=cfg["model"]["h"],
        dropout=cfg["model"]["dropout"]
    )
    if checkpoint_path:
        encoder.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    encoder = encoder.to(device)

    mlm_head = MLMHead(cfg["model"]["d_model"], tokenizer.vocab_size).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(mlm_head.parameters()),
        lr=cfg["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()

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

            if (batch_idx + 1) % 500 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | ", end='')
                print(f"Time Elapsed: {elapsed_time/3600:.2f} hrs ({elapsed_time/60:.2f} mins)")
            
            if (batch_idx + 1) % 10000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"pretrained_encoder_epoch_{epoch+1}_batch_{batch_idx + 1}.pt")
                torch.save(encoder.state_dict(), checkpoint_path)
            
            if (batch_idx + 1) % 10000 == 0:
                drive_path = os.path.join(drive_checkpoint_dir, f"pretrained_encoder_epoch_{epoch+1}_batch_{batch_idx + 1}.pt")
                torch.save(encoder.state_dict(), drive_path)

        avg_loss = total_loss / len(train_loader)

        # Validation loop (every epoch)
        encoder.eval()
        mlm_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask_positions = batch["mask_positions"].to(device)

                mlm_labels = labels.clone()
                mlm_labels[~mask_positions] = -100

                mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)
                encoder_output = encoder(input_ids, mask)
                logits = mlm_head(encoder_output)
                loss = criterion(logits.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Save encoder checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"pretrained_encoder_epoch_{epoch+1}.pt")
        torch.save(encoder.state_dict(), checkpoint_path)
        print(f"Saved encoder checkpoint to {checkpoint_path}")

        drive_path = os.path.join(drive_checkpoint_dir, f"pretrained_encoder_epoch_{epoch+1}.pt")
        torch.save(encoder.state_dict(), drive_path)
        print(f"Saved encoder checkpoint to {drive_path}")