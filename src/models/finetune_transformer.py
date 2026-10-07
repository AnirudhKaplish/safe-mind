"""
finetune_transformer.py
========================
Fine-tunes a transformer model (BERT or RoBERTa) for mental-health
risk classification, per the approved design's "NLP Model Training"
stage - comparing baseline ML (already done in train_baselines.py)
against a transformer-based model.

** RUN THIS ON YOUR OWN MACHINE / COLAB, NOT IN A NETWORK-RESTRICTED
   SANDBOX ** - it needs to download pretrained weights from
   huggingface.co the first time it runs.

Setup (VS Code / local machine):
    pip install torch transformers datasets scikit-learn pandas

Setup (Google Colab):
    !pip install transformers datasets

Usage:
    python finetune_transformer.py --data ../../data/seed_dataset.csv --model bert-base-uncased
    python finetune_transformer.py --data ../../data/seed_dataset.csv --model roberta-base

Once you have the REAL dataset (Kaggle "Depression: Reddit Dataset
(Cleaned)" or Dreaddit), just point --data at that CSV instead -
as long as it has `text` and `label` columns, nothing else changes.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../../data/seed_dataset.csv",
                         help="CSV with `text` and `label` columns")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                         choices=["bert-base-uncased", "roberta-base"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./transformer_output")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have `text` and `label` columns"

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    print("Classes:", list(le.classes_))

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )

    print(f"Loading tokenizer/model: {args.model} (downloads on first run)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(le.classes_)
    )

    train_dataset = TextDataset(train_df["text"], train_df["label_id"], tokenizer)
    test_dataset = TextDataset(test_df["text"], test_df["label_id"], tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,          # standard fine-tuning LR for BERT/RoBERTa
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    print(metrics)

    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    print(f"\nModel saved to {args.output_dir}/final_model")

    # Save metrics for direct comparison against the baseline models
    import json
    with open(f"{args.output_dir}/transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
