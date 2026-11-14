import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv

from src.sentiment.baseline import LABELS
from src.sentiment.experiments import MODEL_PATH as TFIDF_PATH
from src.sentiment.threshold import apply_threshold

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
XLMR_PATH = ROOT / "models" / "xlmr_finetuned"


def choose_model():
    name = os.getenv("SENTIMENT_MODEL", "auto")
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "tfidf"
    return "xlmr" if XLMR_PATH.exists() and torch.cuda.is_available() else "tfidf"


@lru_cache(maxsize=2)
def load(name):
    if name == "tfidf":
        return joblib.load(TFIDF_PATH)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(XLMR_PATH).to(device)
    return {"model": model, "tokenizer": AutoTokenizer.from_pretrained(XLMR_PATH), "device": device}


def predict(texts, name=None):
    name = name or choose_model()
    saved = load(name)
    texts = list(texts)
    if name == "tfidf":
        model = saved["model"]
        proba = model.predict_proba(texts)
        return apply_threshold(proba, list(model.classes_), saved["threshold"]), proba.max(axis=1), name

    from src.sentiment.finetune_xlmr import predict_proba

    proba = predict_proba(saved["model"], saved["tokenizer"], texts, saved["device"])
    return np.array(LABELS, dtype=object)[proba.argmax(axis=1)], proba.max(axis=1), name