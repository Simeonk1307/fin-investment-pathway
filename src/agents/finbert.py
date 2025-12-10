import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pathway as pw

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

class FinBertSentimentAnalyzer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger = logging.getLogger(__name__)
        logger.info(f"Using device: {self.device} for FinBERT Sentiment Analysis")
        self.model.to(self.device)
        self.model.eval()

 
    def analyze_sentiment(self, text: str) -> tuple[float, float, float]:
        # return (0,0,0)
        # logger.info(f"Analyzing sentiment for text{text} of length {len(text)}")
        # def analyze_sentiment(self, text: str) -> float:
        # def analyze_sentiment(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        sentiment_labels = list(self.model.config.id2label.values())
        # logger.info(sentiment_labels)
        # import time
        # time.sleep(5)
        #probability_labels = ["negative", "neutral", "positive"] - change according to sentiment model config
        order_probabilties= {label: prob for label, prob in zip(sentiment_labels, probabilities)}
        predicted_index = probabilities.argmax()
        sentiment_label = sentiment_labels[predicted_index]
        sentiment_score = probabilities[predicted_index] * (1 if sentiment_label == "positive" else -1 if sentiment_label == "negative" else 0)
        # return sentiment_label, float(sentiment_score)
        # return  float(sentiment_score)
        return_order = ["negative", "neutral", "positive"]
        return_probabilties = [order_probabilties[label] for label in return_order]
        # logger.info(f"[FINBERT] Sentiment Analysed {return_probabilties}")
        return tuple(return_probabilties)
    
    def analyze_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results