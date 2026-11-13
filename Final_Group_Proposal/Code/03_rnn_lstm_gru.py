"""
03_rnn_lstm_gru.py
Owner: Group B
Purpose: LSTM / GRU for multi-label classification.
Run: python 03_rnn_lstm_gru.py
Note: Ensure utils.py is in the same directory.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from utils import load_data, evaluate_model, get_label_frequency_vector, print_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
EMBED_DIM = 200
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()[:self.max_len]
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, rnn_type="lstm", pretrained_emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = False  # freeze
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True) \
            if rnn_type == "lstm" else \
            nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        emb = self.embedding(x)                    # (B, L, D)
        out, _ = self.rnn(emb)                     # (B, L, 2H)
        out = out[:, -1, :]                        # last hidden
        out = self.dropout(out)
        return self.fc(out)                        # logits

def build_vocab(texts, max_vocab=50000):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab

def train_rnn(rnn_type="lstm"):
    print(f"=== Training {rnn_type.upper()} ===")
    train_texts, train_labels, mlb = load_data("train")
    test_texts, test_labels, _ = load_data("test")
    freq = get_label_frequency_vector(train_labels)

    vocab = build_vocab(train_texts)
    print(f"Vocab size: {len(vocab)}")

    train_ds = TextDataset(train_texts, train_labels, vocab)
    test_ds = TextDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RNNClassifier(len(vocab), EMBED_DIM, 128, len(mlb.classes_), rnn_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Class weighting: inverse frequency capped
    pos_weight = torch.tensor(np.clip(freq.max() / (freq + 1), 1.0, 10.0), dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d} | Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    y_score = np.vstack(all_scores)

    metrics = evaluate_model(test_labels, y_score, threshold=0.35, label_freq=freq)
    print(f"\n--- {rnn_type.upper()} Results ---")
    print_metrics(metrics)
    return metrics

if __name__ == "__main__":
    train_rnn("lstm")
    train_rnn("gru")