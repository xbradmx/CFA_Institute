"""
DDDS - Vagueness Classifier Inference
======================================
Run the fine-tuned vagueness model on new disclosure passages.

Usage
-----
Single passage (interactive):
    python run_03_vague_predictions.py --model outputs/vagueness_model

Batch from CSV:
    python run_03_vague_predictions.py --model outputs/vagueness_model \
                      --input data/new_passages.csv   \
                      --output data/predictions.csv

The input CSV must have a 'text' column.
Output CSV will be the input CSV with two new columns appended:
    predicted_label  : SPECIFIC or VAGUE
    vague_prob       : model's confidence that the passage is VAGUE (0-1)
"""

import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 512


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[DDDS] Vagueness model loaded from '{model_dir}' on {device}")
    return model, tokenizer, device


def predict_passage(text: str, model, tokenizer, device) -> dict:
    """
    Classify a single passage.

    Returns
    -------
    {
        "label":      "SPECIFIC" or "VAGUE",
        "label_id":   0 or 1,
        "vague_prob": float  (confidence score for VAGUE class)
    }
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs     = torch.softmax(outputs.logits, dim=-1).cpu().squeeze()
    label_id  = int(torch.argmax(probs).item())
    label     = model.config.id2label[label_id]
    vague_prob = float(probs[1].item())

    return {"label": label, "label_id": label_id, "vague_prob": round(vague_prob, 4)}


def predict_batch(df: pd.DataFrame, model, tokenizer, device, batch_size: int = 32) -> pd.DataFrame:
    """
    Classify a DataFrame of passages, returns the DataFrame with predictions appended.
    """
    all_labels     = []
    all_vague_probs = []

    texts = df["text"].astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        probs    = torch.softmax(outputs.logits, dim=-1).cpu()
        label_ids = torch.argmax(probs, dim=-1).tolist()

        all_labels.extend([model.config.id2label[lid] for lid in label_ids])
        all_vague_probs.extend([round(float(probs[j][1].item()), 4) for j in range(len(batch_texts))])

        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} passages", end="\r")

    print()
    df = df.copy()
    df["predicted_label"] = all_labels
    df["vague_prob"]      = all_vague_probs
    return df


def interactive_mode(model, tokenizer, device):
    print("\n[DDDS] Vagueness Classifier  --  Interactive Mode")
    print("  Enter a disclosure passage and get a SPECIFIC / VAGUE prediction.")
    print("  Type 'quit' to exit.\n")

    while True:
        text = input("Passage: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
    
        result = predict_passage(text, model, tokenizer, device)
        bar    = "█" * int(result["vague_prob"] * 30)
        print(f"  Label:      {result['label']}")
        print(f"  Vague prob: {result['vague_prob']:.4f}  |{bar:<30}|")
        print()


def main():
    parser = argparse.ArgumentParser(description="DDDS Vagueness Classifier Inference")
    parser.add_argument("--model",  required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input",  default=None,  help="Input CSV with 'text' column (batch mode)")
    parser.add_argument("--output", default=None,  help="Output CSV path for predictions")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    if args.input:
        df = pd.read_csv(args.input)
        if "text" not in df.columns:
            raise ValueError(f"Input CSV must have a 'text' column. Found: {list(df.columns)}")

        print(f"\n[DDDS] Running batch inference on {len(df)} passages...")
        df_out = predict_batch(df, model, tokenizer, device)

        vague_count    = (df_out["predicted_label"] == "VAGUE").sum()
        specific_count = (df_out["predicted_label"] == "SPECIFIC").sum()
        print(f"\n  Results:  VAGUE={vague_count}  SPECIFIC={specific_count}")
        print(f"  Mean vague probability: {df_out['vague_prob'].mean():.4f}")
        print(f"  STD vague probability: {df_out['vague_prob'].std():.4f}")
        print(f"  Min vague probability: {df_out['vague_prob'].min():.4f}")
        print(f"  Max vague probability: {df_out['vague_prob'].max():.4f}")
        # print(f"  Range vague probability: {df_out['vague_prob'].range():.4f}")


        out_path = args.output or "data/predictions_vagueness.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")

    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
