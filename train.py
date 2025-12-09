import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from models import make_classifier, make_encoder
from tokenizer import CharTokenizer, make_tokenizer
from preprocessing import preprocess_text, filter_short_texts
from helper import load_config


class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], self.max_len)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).unsqueeze(1)
    return mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")


def train(config_path, checkpoint_path=None):
    print("Training")
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["training"]["seed"])

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)

    df = pd.read_csv(cfg["paths"]["data"])
    texts = df["text"].tolist()
    labels = df["generated"].tolist()

    # Preprocess texts
    print(f"Preprocessing {len(texts)} texts...")
    texts = [preprocess_text(text) for text in texts]

    # Filter out very short texts
    original_count = len(texts)
    texts, labels = filter_short_texts(texts, labels, min_length=100)
    filtered_count = original_count - len(texts)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} texts shorter than 100 characters")
    print(f"Final dataset size: {len(texts)} texts")

    tokenizer = make_tokenizer(cfg)
    tokenizer.build_vocab(texts)
    tokenizer.save(cfg["paths"]["vocab"])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=cfg["training"]["seed"]
    )

    train_dataset = EssayDataset(train_texts, train_labels, tokenizer, cfg["training"]["max_seq_len"])
    val_dataset = EssayDataset(val_texts, val_labels, tokenizer, cfg["training"]["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"])

    pretrained_encoder = None
    if checkpoint_path:
        print(f"Loading pretrained encoder from {checkpoint_path}")
        pretrained_encoder = make_encoder(
            vocab=tokenizer.vocab_size,
            N=cfg["model"]["N"],
            d_model=cfg["model"]["d_model"],
            d_ff=cfg["model"]["d_ff"],
            h=cfg["model"]["h"],
            dropout=cfg["model"]["dropout"]
        )
        pretrained_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        pretrained_encoder = pretrained_encoder.to(device)

    model = make_classifier(
        vocab=tokenizer.vocab_size,
        N=cfg["model"]["N"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        h=cfg["model"]["h"],
        dropout=cfg["model"]["dropout"],
        pretrained_encoder=pretrained_encoder
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("Beginning")
    for epoch in range(cfg["training"]["epochs"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["label"].to(device)
            mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)

            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                batch_labels = batch["label"].to(device)
                mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)

                logits = model(input_ids, mask)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"model_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "vocab_size": tokenizer.vocab_size,
            "config": cfg
        }, checkpoint_path)

    # Save final checkpoint
    model_name = Path(config_path).stem
    model_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}.pt")
    torch.save({
        "epoch": cfg["training"]["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "val_acc": history["val_acc"][-1],
        "vocab_size": tokenizer.vocab_size,
        "config": cfg
    }, model_path)

    curves_path = os.path.join(cfg["paths"]["checkpoint_dir"], "training_curves.png")
    plot_curves(history, curves_path)

    print("Training complete.")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained encoder checkpoint")
    args = parser.parse_args()
    train(args.config, args.checkpoint)
