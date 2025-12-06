from src.agents.llm_factory import LLM
from src.agents.agent_state import AgentState
import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

#final agent only output with confidence

class AnalysisReport(BaseModel):
    prediction: str = Field(..., description="Stock price movement prediction: UP, DOWN, or NEUTRAL")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    reason: str = Field(..., description="Brief reason for the prediction in one line. It should be clear and specific")


def final_agent(state:AgentState) -> dict:
    """
    Final agent to consolidate insights and provide final recommendation.
    """
    # logger.info(f"🧠 Processing final analysis through Final Agent {state}")

    try:
        FINAL_PROMPT = f"""
        ROLE: Final Investment Analyst.
        You are a seasoned financial analyst specializing in stock market predictions.
        GOAL: Consolidate insights from previous analyses to provide a final investment recommendation.
        FOCUS: Synthesize information to deliver a clear and concise recommendation.

        INPUT: Previous analyses: 
        news analyst : {state['news_analysis']}
        filings analyst : {state['filings_analysis']}
        social analyst : {state['social_analysis']}
        COMPANY : {state['ticker']}
        market data : {state['market_data']}

        Based on the above analyses, provide:
        1. Final prediction: Will the stock price go UP, DOWN, or stay NEUTRAL?
        2. Confidence level: HIGH, MEDIUM, or LOW
        3. Brief reason (one sentence but clear and specific)

        """
        response=LLM.with_structured_output(AnalysisReport).invoke(FINAL_PROMPT)
        logger.info(f"Final Agent Response: {response} , type  {type(response)}")
        # all analysis agents
        logger.info(f"News Analysis: {state['news_analysis']}")
        logger.info(f"Filings Analysis: {state['filings_analysis']}")
        logger.info(f"Market Analysis: {state['market_data']}")
        logger.info(f"Social Analysis: {state['social_analysis']}")
        logger.info(state['ticker'])
        if isinstance(response, AnalysisReport):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")
        
        expected_keys = {"prediction", "confidence", "reason"}

        if not expected_keys.issubset(result_dict.keys()):
            raise ValueError("Missing keys in the response dictionary")
        return {
            'final_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing final agent response: {e}")
        return {
            "final_analysis": {
                "prediction": "NEUTRAL",
                "confidence": "LOW",
                "reason": "Unable to provide final analysis due to processing error."
        }
        }