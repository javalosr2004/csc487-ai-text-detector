# AI Text Detection Model

A transformer encoder-based model for detecting AI-generated text.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The training data should be a CSV file named `Training_Essay_Data.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `text` | The essay text content |
| `generated` | Label: 0 = human-written, 1 = AI-generated |

## Pretraining 

Pretrain the encoder on BookCorpus using Masked Language Modeling (MLM):

```bash
python pretraining.py --config configs/pretrain.yaml
```

Options:
- `--epochs N` — override number of epochs
- `--max_samples N` — limit dataset size (default: 10M)
- `--checkpoint path/to/checkpoint.pt` — resume from checkpoint

**Features:**
- Deterministic 98/2 train/val split of BookCorpus (configurable via `val_ratio`)
- Validation loss computed every epoch
- Checkpoints saved per epoch and every 10k batches

## Fine Tuning

Train with a pretrained encoder:

```bash
python train.py --config configs/train.yaml --checkpoint checkpoints/encoder.pt
```

This will:
- Load data from `Training_Essay_Data.csv`
- Set random seed to 42 for reproducibility
- Train for 10 epochs with validation every epoch
- Save checkpoints to `checkpoints/`
- Generate training curves (`checkpoints/training_curves.png`)
- Save final model checkpoint (`checkpoints/train.pt`)

## Evaluation

```bash
python eval.py --checkpoint checkpoints/train.pt
```

This will:
- Load the trained model
- Compute accuracy, precision, recall, F1 score
- Generate confusion matrix (`checkpoints/confusion_matrix.png`)

## Inference

```bash
python inference.py --checkpoint checkpoints/train.pt
```

## Configuration

Edit `configs/pretrain.yaml` or `configs/train.yaml` to adjust:

**Model:**
- `d_model`: Hidden dimension (default: 512)
- `d_ff`: Feed-forward dimension (default: 2048)
- `h`: Number of attention heads (default: 8)
- `N`: Number of encoder layers (default: 6)
- `dropout`: Dropout rate (default: 0.1)

**Training:**
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of epochs (default: 10)
- `learning_rate`: Learning rate (default: 0.0001)
- `max_seq_len`: Maximum sequence length (default: 512)
- `seed`: Random seed (default: 42)
- `val_ratio`: Validation split ratio for pretraining (default: 0.02 = 98/2 split)

## Project Structure

```
csc487-ai-text-detector/
├── models/
│   ├── __init__.py
│   └── transformer.py      # Model architecture
├── dataloaders/
│   └── mlm.py              # MLM dataset & BookCorpus loader
├── configs/
│   ├── pretrain.yaml       # Pretraining config
│   └── train.yaml          # Fine-tuning config
├── tokenizer.py            # Character-level tokenizer
├── preprocessing.py        # Preprocessing script
├── pretraining.py          # MLM pretraining script
├── train.py                # Fine-tuning script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
└── README.md
```

## Reproducibility

To reproduce reported metrics:

```bash
pip install -r requirements.txt
python pretraining.py --config configs/pretrain.yaml
python train.py --config configs/train.yaml --checkpoint checkpoints/pretrained_encoder_epoch_5.pt
python eval.py --checkpoint checkpoints/train.pt
```
