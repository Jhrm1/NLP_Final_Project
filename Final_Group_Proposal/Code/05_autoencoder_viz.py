"""
05_autoencoder_viz.py
Owner: Group D
Purpose: Train a text autoencoder on TF-IDF, visualize 2D latent space.
Run: python 05_autoencoder_viz.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from utils import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 2
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def main():
    print("Loading data ...")
    texts, labels, mlb = load_data("train")

    # Use TF-IDF + SVD to get a dense 300-d input (faster & more stable than full TF-IDF)
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    X_sparse = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=300, random_state=SEED)
    X = svd.fit_transform(X_sparse).astype(np.float32)
    print(f"Input shape: {X.shape}")

    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoEncoder(X.shape[1], LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss / len(loader):.4f}")

    # Get latent representations
    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(X).to(DEVICE))
    z = z.cpu().numpy()

    # Pick a frequent label as color (most common among train labels)
    label_freq = labels.sum(axis=0)
    top_idx = np.argmax(label_freq)
    top_label = mlb.classes_[top_idx]
    colors = labels[:, top_idx]  # binary: has top label or not
    print(f"Coloring by label: '{top_label}' (count={int(label_freq[top_idx])})")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(z[colors == 0, 0], z[colors == 0, 1], s=1, alpha=0.3, label=f"Not '{top_label}'")
    plt.scatter(z[colors == 1, 0], z[colors == 1, 1], s=3, alpha=0.7, label=f"'{top_label}'")
    plt.title(f"Autoencoder Latent Space (colored by '{top_label}')")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig("autoencoder_latent.png", dpi=150)
    plt.show()
    print("Saved plot to autoencoder_latent.png")

if __name__ == "__main__":
    main()