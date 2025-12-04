from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

FINBERT_MODEL = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL,
    local_files_only=False,
    ignore_mismatched_sizes=True
)

LABELS = ["positive", "negative", "neutral"]

def get_finbert_sentiment(text):
    if not text or text.strip() == "":
        return "neutral", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]

    score, label_idx = torch.max(probs, dim=-1)
    label = LABELS[label_idx.item()]

    return label, score.item()
