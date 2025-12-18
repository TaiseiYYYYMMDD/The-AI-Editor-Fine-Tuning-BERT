# app/inference.py
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "ai_editor_model_official_9"  # ← ここ1行だけ変える
MODEL_DIR = (BASE_DIR / f"../model_build/{MODEL_NAME}").resolve()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

@torch.no_grad()
def predict(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "EMPTY"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return str(model.config.id2label.get(pred_id, pred_id))
