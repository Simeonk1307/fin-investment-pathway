from src.agents.agent_state import AgentState
import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

class SocialsAnalysisResult(BaseModel):
    prediction: str = Field(..., description="Stock price movement prediction: UP, DOWN, or NEUTRAL")
    sentiment_score: float = Field(..., description="Sentiment score from -1 (very negative) to +1 (very positive)")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    reason: str = Field(..., description="Brief reason for the prediction in one line. It should be clear and specific")

def socials_agent(state:AgentState) -> dict:
    """
    Analyze social media data to extract key insights for investment decision-making.
    """
    # logger.info(f"💬 Processing social media data through Social Analyst Agent {state}")

    try:
        LLM = state['LLM']
        socials_input = state['socials_data']['socials_articles']

        SOCIALS_PROMPT = f"""
        ROLE: Social Media Analyst.
        You are a financial analyst specializing in stock market predictions.
        GOAL: Analyze the provided social media summary to extract key insights relevant for investment decision-making.
        FOCUS: Identify potential risks and opportunities highlighted in social media that could impact the stock's performance.

        COMPANY : {state['ticker']}

        INPUT: {socials_input}(tuple of socials articles)

        Based on this social media data, provide your analysis.
        1. Will the stock price go UP, DOWN, or stay NEUTRAL?
        2. Sentiment score from -1 (very negative) to +1 (very positive)
        3. Confidence level: HIGH, MEDIUM, or LOW
        4. Brief reason (one sentence but clear and specific)

        """
        response=LLM.with_structured_output(SocialsAnalysisResult).invoke(SOCIALS_PROMPT)
        # logger.info(f"Social Analyst Response: {response} , type  {type(response)}")

        if isinstance(response, SocialsAnalysisResult):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")
        
        # Validate keys as per SocialAnalysisResult fields
        # expected_keys = {...}  # Define expected keys based on SocialAnalysisResult
        expected_keys = {"prediction", "sentiment_score", "confidence", "reason"}

        if not expected_keys.issubset(result_dict.keys()):
            raise ValueError("Missing keys in the response dictionary")
        
        # if not expected_keys.issubset(result_dict.keys()):
        #     raise ValueError("Missing keys in the response dictionary")
        # logger.info(f"Socials Analyst Response: {response} , type  {type(response)}")
        # import time
        # time.sleep(5)
        return {
            'socials_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing social agent response: {e}")
        return {
            'socials_analysis': {
            "prediction": "NEUTRAL",
            "sentiment_score": 0.0,
            "confidence": "LOW",
            "reason": "Unable to analyze the socials due to processing error."
        }
            }