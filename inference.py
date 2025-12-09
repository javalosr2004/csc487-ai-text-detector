import argparse
import random

import torch
import pandas as pd

from models import make_classifier
from tokenizer import make_tokenizer
from train import create_mask


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]

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

    return model, tokenizer, cfg["training"]["max_seq_len"]


def predict(model, tokenizer, text, max_len, device):
    tokens = tokenizer.encode(text, max_len)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    mask = create_mask(input_ids, tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        logits = model(input_ids, mask)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()

    return pred, probs[0][pred].item()


def main(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}")

    model, tokenizer, max_len = load_model(checkpoint_path, device)
    print("Model loaded. Enter text to classify (q to quit):\n")

    df = pd.read_csv("data/Training_Essay_Data.csv")
    texts = df["text"].tolist()

    while True:
        
        text = input(">>> ")
        if text.lower() == "q":
            print("Exiting.")
            break
        
        random_text = False
        i = random.randint(0, len(texts)-1)
        if text.lower() == "dataset":
            text = texts[i]
            random_text = True

        if not text.strip():
            continue
        
        if random_text:
            print(f"Randomly Selected Text: {text[:100]}")
            actual_label = "AI-generated" if df.iloc[i, 1] == 1 else "Human-written"
            print(f"Actual: {actual_label}")
        pred, confidence = predict(model, tokenizer, text, max_len, device)
        pred_label = "AI-generated" if pred == 1 else "Human-written"
        print(f"Prediction: {pred_label} (confidence: {confidence:.2%})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)

