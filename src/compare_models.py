"""
Compare m-BERT vs MuRIL performance on the cyberbullying detection task.
Trains both models and produces a comparison table.
"""
import torch
import os
from src.config import MBERT_MODEL, MURIL_MODEL, TRAIN_CSV, IMAGE_DIR, MODEL_DIR, DEVICE
from src.train import train
from src.evaluate import evaluate


def compare_models(csv_path=None, epochs=5):
    """Train and evaluate both m-BERT and MuRIL, then compare results."""
    csv_path = csv_path or TRAIN_CSV

    models_to_compare = {
        'm-BERT': MBERT_MODEL,
        'MuRIL': MURIL_MODEL,
    }

    results = {}

    for name, model_name in models_to_compare.items():
        print("=" * 60)
        print(f"Training: {name} ({model_name})")
        print("=" * 60)

        try:
            # Train
            trained_model = train(
                csv_path=csv_path,
                text_model=model_name,
                epochs=epochs
            )

            # Rename saved model to include model name
            src_path = os.path.join(MODEL_DIR, 'best_model.pth')
            dst_path = os.path.join(MODEL_DIR, f'best_model_{name.lower().replace("-", "_")}.pth')
            if os.path.exists(src_path):
                os.rename(src_path, dst_path)

            print(f"\n--- Evaluating {name} ---")
            evaluate(model_path=dst_path, csv_path=csv_path)

            results[name] = "Completed"

        except Exception as e:
            print(f"ERROR training {name}: {e}")
            results[name] = f"Failed: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("\nDetailed results saved in models/ directory.")


if __name__ == '__main__':
    compare_models(epochs=5)
