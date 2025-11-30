# AI Text Detection Model

A transformer encoder-based model for detecting AI-generated text.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The training data should be a CSV file named `essay_data.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `text` | The essay text content |
| `generated` | Label: 0 = human-written, 1 = AI-generated |

## Training

Train the baseline model:

```bash
python train.py --config configs/baseline.yaml
```

This will:
- Load data from `essay_data.csv`
- Set random seed to 42 for reproducibility
- Train for 10 epochs
- Save checkpoints to `checkpoints/`
- Generate training curves (`checkpoints/training_curves.png`)
- Save final baseline checkpoint (`checkpoints/baseline.pt`)

## Evaluation

Evaluate the baseline model:

```bash
python eval.py --checkpoint checkpoints/baseline.pt
```

This will:
- Load the trained model
- Compute accuracy, precision, recall, F1 score
- Compare against random baseline
- Generate confusion matrix (`checkpoints/confusion_matrix.png`)

## Configuration

Edit `configs/baseline.yaml` to adjust:

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

## Project Structure

```
csc487-ai-text-detector/
├── models/
│   ├── __init__.py
│   └── transformer.py    # Model architecture
├── configs/
│   └── baseline.yaml     # Hyperparameters
├── train.py              # Training script
├── eval.py               # Evaluation script
├── tokenizer.py          # Character-level tokenizer
├── requirements.txt      # Dependencies
└── README.md
```

## Reproducibility

To reproduce reported metrics:

```bash
pip install -r requirements.txt
python train.py --config configs/baseline.yaml
python eval.py --checkpoint checkpoints/baseline.pt
```
