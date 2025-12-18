from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# app/ の1つ上がプロジェクト直下
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / "../model_build/ai_editor_model").resolve()

# 起動時に1回だけロード（毎回ロードすると遅い）
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()

# config.json の id2label を使う（label_map.json不要）
_id2label = _model.config.id2label  # 例: {0:"dokujo-tsushin", ...}

@torch.no_grad()
def predict(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "EMPTY"

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    outputs = _model(**inputs)
    pred_id = int(outputs.logits.argmax(dim=-1).item())

    return _id2label.get(pred_id, f"UNKNOWN({pred_id})")
