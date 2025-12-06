import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pathway as pw

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

        logger = logging.getLogger(__name__)
        # logger.info(f"Analyzing sentiment for text{text} of length {len(text)}")
    # def analyze_sentiment(self, text: str) -> float:
    # def analyze_sentiment(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        sentiment_labels = ["negative", "neutral", "positive"]
        predicted_index = probabilities.argmax()
        sentiment_label = sentiment_labels[predicted_index]
        sentiment_score = probabilities[predicted_index] * (1 if sentiment_label == "positive" else -1 if sentiment_label == "negative" else 0)

        # return sentiment_label, float(sentiment_score)
        # return  float(sentiment_score)
        probabilities = [float(prob) for prob in probabilities]
        return tuple(probabilities)
    
    def analyze_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    

def get_sentiment(title:str="", content:str = "")->tuple[float, float, float]:

    finbert_analyzer = FinBertSentimentAnalyzer()
    text = f"{title}\n {content}"
    
    # logger.info(f"Getting sentiment for text: {text}, length: {len(text)}")
    
    return finbert_analyzer.analyze_sentiment(text)

@pw.udf
def merge(title, content)->str:
    return f"Title: {title}\nContent: {content}"