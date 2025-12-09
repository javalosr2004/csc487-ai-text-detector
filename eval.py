import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from models import make_classifier
from tokenizer import make_tokenizer
from train import EssayDataset, create_mask, set_seed


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "AI"])
    ax.set_yticklabels(["Human", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def random_baseline(labels):
    preds = [random.randint(0, 1) for _ in labels]
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = checkpoint["config"]

    set_seed(cfg["training"]["seed"])

    tokenizer = make_tokenizer(cfg)
    tokenizer.load(cfg["paths"]["vocab"])

    model = make_classifier(
        vocab=checkpoint["vocab_size"],
        N=cfg["model"]["N"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        h=cfg["model"]["h"],
        dropout=cfg["model"]["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    df = pd.read_csv(cfg["paths"]["data"])
    texts = df["text"].tolist()
    labels = df["generated"].tolist()

    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=cfg["training"]["seed"]
    )

    test_dataset = EssayDataset(test_texts, test_labels, tokenizer, cfg["training"]["max_seq_len"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"])

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["label"].to(device)
            mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)

            logits = model(input_ids, mask)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print("\n" + "=" * 40)
    print("MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    rand = random_baseline(all_labels)
    print("\n" + "=" * 40)
    print("RANDOM BASELINE COMPARISON")
    print("=" * 40)
    print(f"Random Accuracy:  {rand['accuracy']:.4f}")
    print(f"Random Precision: {rand['precision']:.4f}")
    print(f"Random Recall:    {rand['recall']:.4f}")
    print(f"Random F1 Score:  {rand['f1']:.4f}")

    print("\n" + "=" * 40)
    print("IMPROVEMENT OVER RANDOM")
    print("=" * 40)
    print(f"Accuracy:  +{(acc - rand['accuracy']) * 100:.2f}%")
    print(f"F1 Score:  +{(f1 - rand['f1']) * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(cfg["paths"]["checkpoint_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    print("\n" + "=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    print(f"True Negatives:  {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives:  {cm[1, 1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.checkpoint)
