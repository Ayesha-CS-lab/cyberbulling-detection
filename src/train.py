"""
Training script for the Cyberbullying Detection model.
Includes train/val split, validation loop, LR scheduler, and early stopping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os
import numpy as np

from src.config import (
    DEVICE, TEXT_MODEL, TRAIN_CSV, IMAGE_DIR, EPOCHS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_LEN,
    VAL_SPLIT, RANDOM_SEED, PATIENCE, MODEL_DIR
)
from src.data.dataset import CyberbullyingDataset
from src.models.fusion_model import CyberbullyingDetector
from src.utils.metrics import compute_metrics


def create_data_loaders(csv_path, image_dir, tokenizer, batch_size, max_len, val_split):
    """Create train and validation DataLoaders with stratified split."""
    full_dataset = CyberbullyingDataset(csv_path, image_dir, tokenizer, max_len=max_len)

    # Create indices for train/val split
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=RANDOM_SEED
    )

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    return train_loader, val_loader


def validate(model, val_loader, criterion, device):
    """Run validation and return average loss + predictions for metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['targets'].to(device)
            context = batch['context'].to(device)

            outputs = model(
                input_ids=ids, attention_mask=mask,
                token_type_ids=None, images=images, context=context
            )
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert logits to predictions (threshold = 0.5)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, threshold=0.5)

    return avg_loss, metrics


def train(
    csv_path=None, image_dir=None, text_model=None,
    epochs=None, batch_size=None, lr=None
):
    """Main training function with validation, scheduling, and early stopping."""
    # Use config defaults if not specified
    csv_path = csv_path or TRAIN_CSV
    image_dir = image_dir or IMAGE_DIR
    text_model = text_model or TEXT_MODEL
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    lr = lr or LEARNING_RATE

    print(f"Device: {DEVICE}")
    print(f"Text Model: {text_model}")
    print(f"Training Data: {csv_path}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print("-" * 50)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model)

    # Data Loaders
    train_loader, val_loader = create_data_loaders(
        csv_path, image_dir, tokenizer, batch_size, MAX_LEN, VAL_SPLIT
    )

    # Model
    model = CyberbullyingDetector(text_model_name=text_model).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2
    )

    # Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            ids = batch['ids'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            labels = batch['targets'].to(DEVICE)
            context = batch['context'].to(DEVICE)

            outputs = model(
                input_ids=ids, attention_mask=mask,
                token_type_ids=None, images=images, context=context
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print()

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")

    return model


if __name__ == '__main__':
    train()
