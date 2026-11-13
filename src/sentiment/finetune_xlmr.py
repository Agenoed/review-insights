"""Fine-tune XLM-RoBERTa for 3-class sentiment on LLM labels + synthetic neutral
reviews. The best epoch is picked on the dev half of the manual labels, the final
evaluation is on the test half (same split as experiments.py).

Usage:
    python -m src.sentiment.finetune_xlmr
    python -m src.sentiment.finetune_xlmr --base cardiffnlp/twitter-xlm-roberta-base-sentiment --name xlmt_finetuned
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.sentiment.baseline import LABELS
from src.sentiment.experiments import CONTROL, load_data, summary, with_synthetic
from src.sentiment.zero_shot import f1_by_lang

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tokenizer, texts, labels, batch_size, shuffle, max_length=128):
    pairs = list(zip(texts, labels))

    def collate(batch):
        encoded = tokenizer([text for text, _ in batch], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([label for _, label in batch])
        return encoded

    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    texts = list(texts)
    loader = make_loader(tokenizer, texts, [0] * len(texts), batch_size, shuffle=False)
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def predict(model, tokenizer, texts, device):
    return np.array(LABELS)[predict_proba(model, tokenizer, texts, device).argmax(axis=1)]


def seconds_per_1000(model, tokenizer, texts, device):
    predict(model, tokenizer, texts[:64], device)
    start = time.perf_counter()
    predict(model, tokenizer, texts, device)
    return round((time.perf_counter() - start) / len(texts) * 1000, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--name", default="xlmr_finetuned")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    dev, test, _, llm, synthetic = load_data()
    texts, labels = with_synthetic(llm["text"], llm["llm_sentiment"], synthetic)
    print(f"Train: {len(texts)} reviews, dev: {len(dev)}, test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    ).to(device)

    loader = make_loader(tokenizer, texts.tolist(), labels.map(LABEL2ID).tolist(), args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    out_dir = MODELS_DIR / args.name
    best_f1, best_epoch = -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        dev_f1 = f1_score(dev["sentiment"], predict(model, tokenizer, dev["text"], device), average="macro")
        print(f"epoch {epoch}: train loss {np.mean(losses):.3f}, dev macro-F1 {dev_f1:.3f}")
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    train_time = time.perf_counter() - start

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    test_pred = predict(model, tokenizer, test["text"], device)
    tfidf = json.loads((METRICS_DIR / "experiments.json").read_text(encoding="utf-8"))["labels"]["LLM labels + synthetic"]

    sample = test["text"].tolist() * 5
    metrics = {
        "base": args.base,
        "best_epoch": best_epoch,
        "dev_macro_f1": round(best_f1, 3),
        "train_size": len(texts),
        "train_time_s": round(train_time, 1),
        "test": {**summary(test["sentiment"], test_pred), "macro_f1_by_lang": f1_by_lang(test, test_pred)},
        "tfidf_llm_synthetic_macro_f1": tfidf["macro_f1"],
        "inference_s_per_1000_gpu": seconds_per_1000(model, tokenizer, sample, device) if device.type == "cuda" else None,
        "inference_s_per_1000_cpu": seconds_per_1000(model.to("cpu"), tokenizer, sample[:300], torch.device("cpu")),
    }

    print(f"\nBest epoch {best_epoch}, trained in {train_time:.0f} s")
    print(f"Test half (n={len(test)}): macro-F1 {metrics['test']['macro_f1']}, "
          f"neutral F1 {metrics['test']['f1_per_class']['neutral']}, by language {metrics['test']['macro_f1_by_lang']}")
    print(f"TF-IDF on the same labels: macro-F1 {metrics['tfidf_llm_synthetic_macro_f1']}")
    print(f"Inference per 1000 reviews: GPU {metrics['inference_s_per_1000_gpu']} s, CPU {metrics['inference_s_per_1000_cpu']} s")

    print("\nControl examples:")
    for text, label in zip(CONTROL, predict(model, tokenizer, CONTROL, torch.device("cpu"))):
        print(f"  {label:<9} {text}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{args.name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved model to {out_dir}, metrics to {METRICS_DIR / (args.name + '.json')}")


if __name__ == "__main__":
    main()