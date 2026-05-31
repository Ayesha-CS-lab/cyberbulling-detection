"""
Evaluation script: loads a trained model, runs it on the test set,
and prints a full classification report with confusion matrices.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os

from src.config import (
    DEVICE, TEXT_MODEL, TRAIN_CSV, IMAGE_DIR,
    BATCH_SIZE, MAX_LEN, VAL_SPLIT, TEST_SPLIT,
    RANDOM_SEED, MODEL_DIR
)
from src.data.dataset import CyberbullyingDataset
from src.models.fusion_model import CyberbullyingDetector
from src.utils.metrics import compute_metrics, get_classification_report, get_confusion_matrices


def evaluate(model_path=None, csv_path=None):
    """Evaluate a trained model on the held-out test set."""
    csv_path = csv_path or TRAIN_CSV
    model_path = model_path or os.path.join(MODEL_DIR, 'best_model.pth')

    print(f"Device: {DEVICE}")
    print(f"Model: {model_path}")
    print(f"Data: {csv_path}")
    print("-" * 50)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    # Load full dataset and extract test split
    full_dataset = CyberbullyingDataset(csv_path, IMAGE_DIR, tokenizer, max_len=MAX_LEN)

    indices = list(range(len(full_dataset)))
    # First split off the test set
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )

    test_subset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test samples: {len(test_subset)}")

    # Load model
    model = CyberbullyingDetector(text_model_name=TEXT_MODEL).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"WARNING: Model file not found at {model_path}. Using untrained model.")

    model.eval()

    # Run predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['ids'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            context = batch['context'].to(DEVICE)
            labels = batch['targets']

            outputs = model(
                input_ids=ids, attention_mask=mask,
                token_type_ids=None, images=images, context=context
            )

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Classification Report
    report = get_classification_report(all_labels, all_preds)
    print(report)

    # Confusion Matrices
    matrices = get_confusion_matrices(all_labels, all_preds)
    for name, matrix in matrices.items():
        print(f"\nConfusion Matrix - {name}:")
        print(matrix)

    # Save report to file
    report_path = os.path.join(MODEL_DIR, 'evaluation_report.txt')
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
        for name, matrix in matrices.items():
            f.write(f"\nConfusion Matrix - {name}:\n")
            f.write(str(matrix) + "\n")
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    evaluate()
