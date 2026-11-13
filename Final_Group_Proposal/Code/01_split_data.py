"""
01_split_data.py
Owner: Group A
Purpose: Download SciX_UAT_keywordcds, create a fixed split, save to data_cache/.
Run: python 01_split_data.py
"""
import os
import json
import numpy as np
from datasets import load_dataset
from utils import _write_jsonl, _normalize_record, LOCAL_CACHE_DIR

HF_DATASET_NAME = "adsabs/SciX_UAT_keywords"
SEED = 42

def main():
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    print(f"Loading {HF_DATASET_NAME} ...")
    ds = load_dataset(HF_DATASET_NAME, split="train")
    records = [dict(row) for row in ds]
    print(f"Total records: {len(records)}")

    # Filter: ensure text and labels exist
    valid = []
    for rec in records:
        text, labels = _normalize_record(rec)
        if text and labels:
            # Rebuild a clean dict
            valid.append({
                "text": text,
                "labels": labels,
                "bibcode": rec.get("bibcode", ""),
            })
    print(f"Valid records after filtering: {len(valid)}")

    # Deterministic split: 85% train, 5% val, 10% test
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(valid))
    n = len(valid)
    n_train = int(0.85 * n)
    n_val = int(0.05 * n)

    train = [valid[i] for i in idx[:n_train]]
    val = [valid[i] for i in idx[n_train:n_train + n_val]]
    test = [valid[i] for i in idx[n_train + n_val:]]

    for name, recs in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(LOCAL_CACHE_DIR, f"{name}.jsonl")
        _write_jsonl(path, recs)
        print(f"Saved {len(recs)} records to {path}")

    # Also persist label list for reference
    all_labels = sorted({l for rec in train for l in rec["labels"]})
    with open(os.path.join(LOCAL_CACHE_DIR, "label_list.json"), "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"Unique labels in train: {len(all_labels)}")

if __name__ == "__main__":
    main()