"""
DDDS - Vagueness Classifier Fine-Tuning Pipeline
=================================================
Model:  ProsusAI/finbert  (base)
Task:   Binary classification  ->  0 = Specific  |  1 = Vague
Framework: PyTorch + HuggingFace Transformers

Data contract
-------------
Expects a CSV with at minimum two columns:
    text   : str  - the 512-token (or shorter) passage
    label  : int  - 0 (specific) or 1 (vague)

Recommended split: 70% train / 15% val / 15% test
If you only have train + val, set TEST_CSV = None and the held-out
evaluation step is skipped.

Usage
-----
    python train.py                          # uses paths defined below
    python train.py --train data/train.csv \\
                   --val   data/val.csv   \\
                   --test  data/test.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG  -  edit these paths before running
# ---------------------------------------------------------------------------
BASE_MODEL      = "ProsusAI/finbert"          # FinBERT base checkpoint
TRAIN_CSV       = "data/training/vagueness/train.csv"
VAL_CSV         = "data/training/vagueness/val.csv"
TEST_CSV        = "data/training/vagueness/test.csv"            # held-out test set (or None)
OUTPUT_DIR      = "outputs/vagueness_model"   # where the fine-tuned model is saved
MAX_LENGTH      = 512                         # FinBERT max token length
BATCH_SIZE      = 16                          # reduce to 8 if OOM on GPU
LEARNING_RATE   = 2e-5
NUM_EPOCHS      = 5
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01
SEED            = 42

# Label mapping
ID2LABEL = {0: "SPECIFIC", 1: "VAGUE"}
LABEL2ID = {"SPECIFIC": 0, "VAGUE": 1}
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DisclosureDataset(Dataset):
    """
    Wraps a pandas DataFrame into a HuggingFace-compatible PyTorch Dataset.

    Expects the DataFrame to have:
        text  : str column
        label : int column (0 or 1)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Validate required columns
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV at '{path}' is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Coerce labels to int and validate values
    df["label"] = df["label"].astype(int)
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        raise ValueError(
            f"Labels must be 0 or 1. Found unexpected values: {invalid['label'].unique()}"
        )

    # Drop rows with missing text
    before = len(df)
    df = df.dropna(subset=["text"])
    dropped = before - len(df)
    if dropped:
        print(f"  [!] Dropped {dropped} rows with missing text")

    df["text"] = df["text"].astype(str).str.strip()
    return df


def compute_metrics(eval_pred):
    """
    Called by HuggingFace Trainer after each evaluation step.
    Returns accuracy, macro F1, and per-class precision/recall/F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    # Per-class metrics
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy":          acc,
        "macro_f1":          f1,
        "macro_precision":   precision,
        "macro_recall":      recall,
        "f1_specific":       f1_cls[0],
        "f1_vague":          f1_cls[1],
        "precision_specific": p_cls[0],
        "precision_vague":   p_cls[1],
        "recall_specific":   r_cls[0],
        "recall_vague":      r_cls[1],
    }


def print_evaluation_report(model, dataset, tokenizer, split_name: str):
    """
    Runs full classification_report and confusion matrix on a dataset split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds  = []
    all_labels = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    print(f"\n{'='*60}")
    print(f"  Evaluation Report  --  {split_name}")
    print(f"{'='*60}")
    print(classification_report(
        all_labels, all_preds,
        target_names=["SPECIFIC (0)", "VAGUE (1)"],
        digits=4
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"                Pred SPECIFIC  Pred VAGUE")
    print(f"  True SPECIFIC    {cm[0][0]:>6}         {cm[0][1]:>6}")
    print(f"  True VAGUE       {cm[1][0]:>6}         {cm[1][1]:>6}")


def main(train_csv: str, val_csv: str, test_csv: str | None):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DDDS] Vagueness Classifier Fine-Tuning")
    print(f"  Device:     {device}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Train data: {train_csv}")
    print(f"  Val data:   {val_csv}")
    print(f"  Test data:  {test_csv or 'None (skipping held-out eval)'}\n")

    # --- Load data ---
    print("[1/5] Loading data...")
    train_df = load_data(train_csv)
    val_df   = load_data(val_csv)
    print(f"  Train: {len(train_df)} rows  "
          f"(vague: {train_df['label'].sum()}, specific: {(train_df['label']==0).sum()})")
    print(f"  Val:   {len(val_df)} rows  "
          f"(vague: {val_df['label'].sum()}, specific: {(val_df['label']==0).sum()})")

    if test_csv:
        test_df = load_data(test_csv)
        print(f"  Test:  {len(test_df)} rows  "
              f"(vague: {test_df['label'].sum()}, specific: {(test_df['label']==0).sum()})")

    # --- Tokenizer ---
    print("\n[2/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_dataset = DisclosureDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = DisclosureDataset(val_df,   tokenizer, MAX_LENGTH)
    if test_csv:
        test_dataset = DisclosureDataset(test_df, tokenizer, MAX_LENGTH)

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # replaces FinBERT's 3-class head with binary
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # --- Class weights (handles imbalanced data) ---
    # If your dataset is balanced you can remove this; it does no harm if kept.
    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / label_counts[i] for i in range(2)], dtype=torch.float
    )
    class_weights = class_weights / class_weights.sum()
    print(f"  Class weights: SPECIFIC={class_weights[0]:.4f}, VAGUE={class_weights[1]:.4f}")

    # --- Training arguments ---
    print("\n[3/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "macro_f1",
        greater_is_better           = True,
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = SEED,
        fp16                        = torch.cuda.is_available(),  # mixed precision on GPU
        dataloader_num_workers      = 0,
    )

    # --- Trainer ---
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # --- Train ---
    print("\n[4/5] Training...")
    trainer.train()

    # --- Save ---
    print(f"\n[5/5] Saving model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("  Model and tokenizer saved.")

    # --- Validation report ---
    print_evaluation_report(model, val_dataset, tokenizer, "Validation Set")

    # --- Held-out test report ---
    if test_csv:
        print_evaluation_report(model, test_dataset, tokenizer, "Test Set (Held-Out)")

    print("\n[Done] Fine-tuning complete.")
    print(f"  Load your model with:\n"
          f"    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n"
          f"    model     = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')\n"
          f"    tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT vagueness classifier")
    parser.add_argument("--train", default=TRAIN_CSV,  help="Path to training CSV")
    parser.add_argument("--val",   default=VAL_CSV,    help="Path to validation CSV")
    parser.add_argument("--test",  default=TEST_CSV,   help="Path to test CSV (optional)")
    args = parser.parse_args()

    main(args.train, args.val, args.test)
