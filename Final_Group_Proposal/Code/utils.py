"""
utils.py
Shared data loading + evaluation utilities for the AstroConcepts project.

Owner: Group A 
Consumers: B (RNN), C (Transformer), D (Autoencoder + XAI)

Usage:
    from utils import load_data, evaluate_model, get_label_frequency_bins

    train_texts, train_labels, mlb = load_data("train")
    test_texts,  test_labels,  _   = load_data("test")

    y_score = model.predict_proba(test_texts)   # shape (n, n_labels)
    metrics = evaluate_model(test_labels, y_score, threshold=0.35)
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_DATASET_NAME = "adsabs/SciX_UAT_keywords"
LOCAL_CACHE_DIR = os.environ.get("ASTROCONCEPTS_CACHE", "./data_cache")
SPLIT_FILES = {
    "train": "train.jsonl",
    "test":  "test.jsonl",
    "val":   "val.jsonl",   # optional; if absent we carve val out of train
}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Low-level I/O
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize_record(rec: dict) -> Tuple[str, List[str]]:
    """
    Accept a few possible field namings from different dataset dumps.
    Returns (text, list_of_labels).
    """
    title = rec.get("title") or rec.get("Title") or ""
    abstract = rec.get("abstract") or rec.get("Abstract") or rec.get("summary") or ""
    text = (rec.get("text") or "").strip()
    if not text:
        text = (title.strip() + " " + abstract.strip()).strip()

    raw_labels = (
        rec.get("uat_concepts")
        or rec.get("verified_uat_labels")
        or rec.get("concepts")
        or rec.get("labels")
        or rec.get("keywords")
        or []
    )
    # Some dumps store labels as a comma/semicolon-separated string
    if isinstance(raw_labels, str):
        raw_labels = re.split(r"[;,|]", raw_labels)
    labels = [str(l).strip() for l in raw_labels if str(l).strip()]
    return text, labels


# ---------------------------------------------------------------------------
# 2. Public loader
# ---------------------------------------------------------------------------
def load_data(
    split: str = "train",
    cache_dir: str = LOCAL_CACHE_DIR,
    return_mlb: bool = True,
) -> Tuple[List[str], np.ndarray, MultiLabelBinarizer]:
    """
    Load one split of the AstroConcepts corpus.

    Parameters
    ----------
    split : {"train", "test", "val"}
    cache_dir : directory holding pre-split .jsonl files (recommended workflow)
    return_mlb : if True, also return the MultiLabelBinarizer fitted on TRAIN

    Returns
    -------
    texts  : list[str]                — "title + ' ' + abstract"
    labels : np.ndarray (n, n_labels) — multi-hot 0/1 matrix
    mlb    : MultiLabelBinarizer      — fitted on train label space (None if return_mlb=False)

    Behaviour
    ---------
    * If `cache_dir/<split>.jsonl` exists, it is used (this is how Group A
      guarantees B/C/D see identical splits — run `01_split_data.py` once).
    * Otherwise, the function falls back to downloading from Hugging Face
      and, for "train", performs a deterministic 90/10 train/val split
      with RANDOM_SEED.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, SPLIT_FILES[split])

    if os.path.exists(path):
        records = _read_jsonl(path)
    else:
        # ---- Fallback: download from Hugging Face -------------------------
        try:
            from datasets import load_dataset  # lazy import
        except ImportError as e:
            raise ImportError(
                "Please `pip install datasets` or run 01_split_data.py first "
                "to cache the data locally."
            ) from e

        ds = load_dataset(HF_DATASET_NAME)["train"]
        records = [dict(row) for row in ds]

        # Deterministic train / val / test split if not yet cached
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.permutation(len(records))
        n = len(records)
        n_train = int(0.85 * n)
        n_val = int(0.05 * n)

        split_map = {
            "train": [records[i] for i in idx[:n_train]],
            "val":   [records[i] for i in idx[n_train:n_train + n_val]],
            "test":  [records[i] for i in idx[n_train + n_val:]],
        }
        for s, recs in split_map.items():
            _write_jsonl(os.path.join(cache_dir, SPLIT_FILES[s]), recs)
        records = split_map[split]

    texts, label_lists = [], []
    for rec in records:
        t, labs = _normalize_record(rec)
        if t and labs:
            texts.append(t)
            label_lists.append(labs)

    if return_mlb:
        mlb = _get_or_fit_mlb(cache_dir)
        labels = mlb.transform(label_lists)
    else:
        labels = label_lists  # return raw lists
        mlb = None

    return texts, labels, mlb


# ---------------------------------------------------------------------------
# 3. Label space: fit once on train, persist for reuse
# ---------------------------------------------------------------------------
def _get_or_fit_mlb(cache_dir: str) -> MultiLabelBinarizer:
    """Fit the label binarizer on TRAIN only and cache it as .npy + json."""
    mlb_path = os.path.join(cache_dir, "mlb_classes.json")
    if os.path.exists(mlb_path):
        with open(mlb_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        mlb = MultiLabelBinarizer(classes=classes)
        mlb.fit([classes])  # rebuild internal state
        return mlb

    train_path = os.path.join(cache_dir, SPLIT_FILES["train"])
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Train split not found. Run 01_split_data.py or load_data('train') first."
        )
    recs = _read_jsonl(train_path)
    label_lists = [_normalize_record(r)[1] for r in recs]
    mlb = MultiLabelBinarizer()
    mlb.fit(label_lists)
    with open(mlb_path, "w", encoding="utf-8") as f:
        json.dump(list(mlb.classes_), f)
    return mlb


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.35,
    top_k: Tuple[int, ...] = (1, 3, 5),
    label_freq: np.ndarray | None = None,
) -> dict:
    """
    Unified evaluation for multi-label classification.

    Parameters
    ----------
    y_true   : (n, L) binary ground-truth matrix
    y_score  : (n, L) float array — probabilities or decision scores
    y_pred   : (n, L) binary predictions. If None, derived by thresholding y_score.
    threshold: decision threshold used when y_pred is None
    top_k    : tuple of k values for P@k / R@k
    label_freq: optional (L,) array of train-time positive counts. If given,
                metrics are additionally reported per Head/Torso/Tail bin.

    Returns
    -------
    dict with keys:
        accuracy, micro_f1, macro_f1, weighted_f1,
        micro_precision, micro_recall,
        macro_precision, macro_recall,
        precision@{k}, recall@{k}  for each k in top_k,
        per_label_f1 (np.ndarray, length L),
        frequency_bins (dict) — only if label_freq is provided
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    out = {}

    # ---- Set-based metrics (exact match / micro / macro) ----------------
    out["accuracy"]         = float(accuracy_score(y_true, y_pred))
    out["micro_precision"]  = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    out["micro_recall"]     = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    out["micro_f1"]         = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_precision"]  = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_recall"]     = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_f1"]         = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["weighted_f1"]      = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ---- Per-label F1 (useful for frequency-stratified analysis) --------
    _, _, per_label_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    out["per_label_f1"] = per_label_f1

    # ---- Ranking metrics P@k / R@k -------------------------------------
    for k in top_k:
        p, r = _precision_recall_at_k(y_true, y_score, k)
        out[f"precision@{k}"] = p
        out[f"recall@{k}"]    = r

    # ---- Frequency-stratified breakdown (Head/Torso/Tail) --------------
    if label_freq is not None:
        out["frequency_bins"] = _stratify_by_frequency(
            y_true, y_pred, per_label_f1, np.asarray(label_freq)
        )

    return out


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> Tuple[float, float]:
    """Standard multi-label P@k / R@k, averaged over samples."""
    n = y_true.shape[0]
    k = min(k, y_true.shape[1])
    # Top-k indices per row
    topk_idx = np.argsort(-y_score, axis=1)[:, :k]

    p_list, r_list = [], []
    for i in range(n):
        true_pos = np.where(y_true[i] == 1)[0]
        if len(true_pos) == 0:
            continue
        pred_pos = topk_idx[i]
        hits = len(set(pred_pos) & set(true_pos))
        p_list.append(hits / k)
        r_list.append(hits / len(true_pos))
    return (float(np.mean(p_list)) if p_list else 0.0,
            float(np.mean(r_list)) if r_list else 0.0)


def _stratify_by_frequency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_label_f1: np.ndarray,
    label_freq: np.ndarray,
) -> dict:
    """
    Split labels into Head (>500), Torso (50–500), Tail (<50) bins
    based on TRAIN frequency, and report mean per-label F1 and support.
    """
    bins = {
        "Head (>500)":   label_freq > 500,
        "Torso (50–500)": (label_freq >= 50) & (label_freq <= 500),
        "Tail (<50)":    label_freq < 50,
    }
    result = {}
    for name, mask in bins.items():
        if mask.sum() == 0:
            result[name] = {"n_labels": 0}
            continue
        f1_bin = per_label_f1[mask]
        # also report global micro-F1 restricted to columns in this bin
        micro_f1 = float(
            f1_score(y_true[:, mask], y_pred[:, mask], average="micro", zero_division=0)
        )
        result[name] = {
            "n_labels":     int(mask.sum()),
            "mean_label_f1": float(np.mean(f1_bin)),
            "micro_f1":      micro_f1,
        }
    return result


# ---------------------------------------------------------------------------
# 5. Convenience: label frequency vector
# ---------------------------------------------------------------------------
def get_label_frequency_bins(
    y_train: np.ndarray,
    thresholds: Tuple[int, int] = (500, 50),
) -> dict:
    """
    Return a dict mapping bin name -> column indices, computed from TRAIN labels.

    Example
    -------
    >>> freq = get_label_frequency_bins(train_labels)
    >>> freq["Head (>500)"]  # array of column indices
    """
    head_thr, tail_thr = thresholds
    freq = y_train.sum(axis=0)
    return {
        f"Head (>{head_thr})":          np.where(freq > head_thr)[0],
        f"Torso ({tail_thr}–{head_thr})": np.where((freq >= tail_thr) & (freq <= head_thr))[0],
        f"Tail (<{tail_thr})":          np.where(freq < tail_thr)[0],
        "_freq": freq,
    }


def get_label_frequency_vector(y_train: np.ndarray) -> np.ndarray:
    """Just the per-label positive count from TRAIN (for evaluate_model(label_freq=...))."""
    return y_train.sum(axis=0)


# ---------------------------------------------------------------------------
# 6. Pretty-print helper 
# ---------------------------------------------------------------------------
def print_metrics(metrics: dict) -> None:
    print("=" * 60)
    print(f"{'Metric':<28}{'Value':>12}")
    print("-" * 60)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk:<24}{vv}")
        elif isinstance(v, np.ndarray):
            print(f"{k:<28}{'<array len=' + str(len(v)) + '>':>12}")
        else:
            print(f"{k:<28}{v:>12.4f}" if isinstance(v, float) else f"{k:<28}{v:>12}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny in-memory test that does not require the real dataset
    from sklearn.preprocessing import MultiLabelBinarizer as _MLB

    _mlb = _MLB()
    _y_true = _mlb.fit_transform([["A", "B"], ["B"], ["C"]])
    _y_score = np.array([[0.9, 0.7, 0.1],
                         [0.2, 0.8, 0.3],
                         [0.1, 0.2, 0.6]])
    _freq = np.array([500, 50, 10])  # fake train frequencies → Head/Torso/Tail

    m = evaluate_model(_y_true, _y_score, threshold=0.5, label_freq=_freq)
    print_metrics(m)