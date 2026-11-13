"""
04_bert_finetune.py
Owner: Group C
Purpose: Fine-tune astroBERT / BERT for multi-label classification.
Run: python 04_bert_finetune.py
Note: Requires GPU. Adjust batch_size if OOM.
"""
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import f1_score
from utils import load_data, evaluate_model, get_label_frequency_vector, print_metrics

MODEL_NAME = "adsabs/astroBERT"  # fallback: "bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 8
SEED = 42

os.environ["WANDB_DISABLED"] = "true"

class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.35).astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
    }

def main():
    train_texts, train_labels, mlb = load_data("train")
    val_texts, val_labels, _ = load_data("val")
    test_texts, test_labels, _ = load_data("test")
    freq = get_label_frequency_vector(train_labels)
    num_labels = len(mlb.classes_)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification"
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = AstroDataset(train_texts, train_labels, tokenizer)
    val_ds = AstroDataset(val_texts, val_labels, tokenizer)
    test_ds = AstroDataset(test_texts, test_labels, tokenizer)

    args = TrainingArguments(
        output_dir="./bert_ckpt",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.1,
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Fine-tuning ...")
    trainer.train()

    # Final evaluation on test set
    print("\n=== Test Set Evaluation ===")
    preds = trainer.predict(test_ds)
    probs = 1 / (1 + np.exp(-preds.predictions))
    metrics = evaluate_model(test_labels, probs, threshold=0.35, label_freq=freq)
    print_metrics(metrics)

    # Save the model for SHAP/LIME use
    trainer.save_model("./best_astrobert")
    print("Model saved to ./best_astrobert")

if __name__ == "__main__":
    main()