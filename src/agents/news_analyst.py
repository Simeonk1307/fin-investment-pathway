from src.agents.agent_state import AgentState
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

class NewsAnalysisResult(BaseModel):
    prediction: str = Field(..., description="Stock price movement prediction: UP, DOWN, or NEUTRAL")
    sentiment_score: float = Field(..., description="Sentiment score from -1 (very negative) to +1 (very positive)")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    reason: str = Field(..., description="Brief reason for the prediction in one line. It should be clear and specific")


def news_agent(state:AgentState) -> dict:
    """
    Analyze news summary to extract key insights for investment decision-making.
    """
    # logger.info(f"📰 Processing news article through News Analyst Agent {state}")

    try:
        LLM = state['LLM']
        # news_text =f"Headline: {state['news']['headline']}\nDescription: {state['news']['description']}"
        news_text=state['news_data']['news_articles']
        NEWS_PROMPT = f"""
        ROLE: News Analyst.
        You are a financial analyst specializing in stock market predictions.
        GOAL: Analyze the provided news summary to extract key insights relevant for investment decision-making.
        FOCUS: Identify potential risks and opportunities highlighted in the news that could impact the stock's performance.
        
        COMPANY : {state['ticker']}

        INPUT: {news_text}(tuple of news articles over past month)

        Based on this news, predict:
        1. Will the stock price go UP, DOWN, or stay NEUTRAL?
        2. Sentiment score from -1 (very negative) to +1 (very positive)
        3. Confidence level: HIGH, MEDIUM, or LOW
        4. Brief reason (one sentence but clear and specific)

        """
        # response = LLM.invoke(NEWS_PROMPT)
        response=LLM.with_structured_output(NewsAnalysisResult).invoke(NEWS_PROMPT)
        
        #check whether response.content is dict or str and all keys are there 
        if isinstance(response, NewsAnalysisResult):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")
        
        # Validate keys
        expected_keys = {"prediction", "sentiment_score", "confidence", "reason"}

        if not expected_keys.issubset(result_dict.keys()):
            raise ValueError("Missing keys in the response dictionary")
        
        # logger.info(f"News Analyst Response: {response} , type  {type(response)}")
        logger.info(f"[AGENT PIPELINE] News analysis received : {result_dict}")
        return {
            'news_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing news analyst response: {e}")

        return {'news_analysis': {
            "prediction": "NEUTRAL",
            "sentiment_score": 0.0,
            "confidence": "LOW",
            "reason": "Unable to analyze the news due to processing error."
        }
        }

