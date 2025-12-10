from src.agents.agent_state import AgentState
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

#final agent only output with confidence

class AnalysisReport(BaseModel):
    prediction: str = Field(..., description="Stock price movement prediction: UP, DOWN, or NEUTRAL")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    reason: str = Field(..., description="Brief reason for the prediction in one line. It should be clear and specific")
    strategy: str = Field(..., description="Action strategy: BUY, SELL or HOLD")

def final_agent(state:AgentState) -> dict:
    """
    Final agent to consolidate insights and provide final recommendation.
    """
    # logger.info(f"🧠 Processing final analysis through Final Agent {state}")

    try:
        LLM = state['LLM']
        FINAL_PROMPT = f"""
        ROLE: Final Investment Analyst.
        You are a seasoned financial analyst specializing in stock market predictions.
        GOAL: Consolidate insights from previous analyses to provide a final investment recommendation.
        FOCUS: Synthesize information to deliver a clear and concise recommendation.

        INPUT: Previous analyses: 
        news analyst : {state['news_analysis']}

        news_sentimental_scores :[ {state['news_data'].get('news_sentiment_scores',{})} in the format of (negative, neutral, positive)
        Time-weighted probability scores from FinBERT analysis, with recent articles weighted more heavily. ]
        
        filings analyst : {state['filings_analysis']}

        social analyst : {state['socials_analysis']}

        COMPANY : {state['ticker']}

        market data : {state['market_data']}[tuple of (current_price:float, volume:int, timestamp:int, predicted_price:float)]

        Based on the above analyses, provide:
        1. Final prediction: Will the stock price go UP, DOWN, or stay NEUTRAL?
        2. Confidence level: HIGH, MEDIUM, or LOW
        3. Brief reason (one sentence but clear and specific)

        """
        response=LLM.with_structured_output(AnalysisReport).invoke(FINAL_PROMPT)

        # logger.info(f"Final Agent Response: {response} , type  {type(response)}")

        # # all analysis agents
        # logger.info(f"News Analysis: {state['news_analysis']}")
        # logger.info(f"Filings Analysis: {state['filings_analysis']}")
        # logger.info(f"Market Analysis: {state['market_data']}")
        # logger.info(f"Social Analysis: {state['socials_analysis']}")
        # logger.info(state['ticker'])
       
       
        if isinstance(response, AnalysisReport):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")
        
        expected_keys = {"prediction", "confidence", "reason", "strategy"}

        if not expected_keys.issubset(result_dict.keys()):
            raise ValueError("Missing keys in the response dictionary")
        
        logger.info(f"[AGENT PIPELINE] Final decision for {state['ticker']} received : {result_dict}")
        
        return {
            'final_analysis': result_dict
        }
    
    except Exception as e:
        return {
            "final_analysis": {
                "prediction": "NEUTRAL",
                "confidence": "LOW",
                "reason": "Unable to provide final analysis due to processing error.",
                "strategy": "Unable to determine strategy - Best to HOLD or SELL. "
        }
        }