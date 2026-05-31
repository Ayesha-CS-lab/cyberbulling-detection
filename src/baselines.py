"""
Baseline models for comparison against the multimodal fusion model.
Implements SVM + TF-IDF and LSTM baselines.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import os

from src.config import TRAIN_CSV, RANDOM_SEED, TEST_SPLIT
from src.data.preprocessing import TextPreprocessor


LABEL_NAMES = ['Aggression', 'Repetition', 'Intent']


def load_and_preprocess(csv_path):
    """Load data and clean text."""
    df = pd.read_csv(csv_path)
    preprocessor = TextPreprocessor()

    texts = []
    labels = []

    for _, row in df.iterrows():
        text = str(row['text_content'])
        lang = row.get('language', 'en')
        clean = preprocessor.preprocess(text, lang=lang)
        texts.append(clean)
        labels.append([
            int(row.get('aggression', 0)),
            int(row.get('repetition', 0)),
            int(row.get('intent', 0))
        ])

    return texts, np.array(labels)


def svm_baseline(csv_path=None):
    """
    SVM + TF-IDF baseline (text-only, no images).
    Uses One-vs-Rest strategy for multi-label classification.
    """
    csv_path = csv_path or TRAIN_CSV
    print("=" * 60)
    print("BASELINE: SVM + TF-IDF")
    print("=" * 60)

    texts, labels = load_and_preprocess(csv_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )

    # TF-IDF Features
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train SVM (One-vs-Rest for multi-label)
    print("Training SVM...")
    svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    svm.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = svm.predict(X_test_tfidf)

    # Results
    print("\n--- Results ---")
    for i, name in enumerate(LABEL_NAMES):
        print(f"\n{name}:")
        print(classification_report(
            y_test[:, i], y_pred[:, i],
            target_names=['Non-Bullying', 'Bullying'],
            zero_division=0
        ))

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro):       {f1_macro:.4f}")

    return {'accuracy': accuracy, 'f1_macro': f1_macro}


def lstm_baseline(csv_path=None):
    """
    Simple LSTM baseline (text-only, no images).
    Uses a single-layer LSTM with word embeddings.
    """
    csv_path = csv_path or TRAIN_CSV
    print("\n" + "=" * 60)
    print("BASELINE: LSTM")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch required for LSTM baseline")
        return None

    texts, labels = load_and_preprocess(csv_path)

    # Simple tokenization (word-level)
    from collections import Counter
    word_freq = Counter()
    for text in texts:
        word_freq.update(text.lower().split())

    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_freq.most_common(10000))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    max_len = 128

    def encode_text(text):
        tokens = text.lower().split()[:max_len]
        encoded = [vocab.get(w, 1) for w in tokens]
        # Pad
        encoded += [0] * (max_len - len(encoded))
        return encoded

    X_encoded = np.array([encode_text(t) for t in texts])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float)
    X_test_t = torch.tensor(X_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.float)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # LSTM Model
    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            emb = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(emb)
            # Use last hidden state from both directions
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            hidden = self.dropout(hidden)
            return self.fc(hidden)

    model = SimpleLSTM(vocab_size=len(vocab) + 2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    print("Training LSTM...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        y_pred = (torch.sigmoid(outputs).cpu().numpy() >= 0.5).astype(int)

    y_test_np = y_test

    print("\n--- Results ---")
    for i, name in enumerate(LABEL_NAMES):
        print(f"\n{name}:")
        print(classification_report(
            y_test_np[:, i], y_pred[:, i],
            target_names=['Non-Bullying', 'Bullying'],
            zero_division=0
        ))

    accuracy = accuracy_score(y_test_np, y_pred)
    f1_macro = f1_score(y_test_np, y_pred, average='macro', zero_division=0)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro):       {f1_macro:.4f}")

    return {'accuracy': accuracy, 'f1_macro': f1_macro}


def run_all_baselines(csv_path=None):
    """Run all baselines and print comparison table."""
    csv_path = csv_path or TRAIN_CSV

    results = {}
    results['SVM + TF-IDF'] = svm_baseline(csv_path)
    results['LSTM'] = lstm_baseline(csv_path)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 (Macro)':<12}")
    print("-" * 44)
    for name, metrics in results.items():
        if metrics:
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f}")
        else:
            print(f"{name:<20} {'Failed':<12} {'Failed':<12}")

    print("\nNote: These are text-only baselines. The fusion model")
    print("additionally uses image and context features.")


if __name__ == '__main__':
    run_all_baselines()
