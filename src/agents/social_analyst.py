from src.agents.llm_factory import LLM
from src.agents.agent_state import AgentState
import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

class SocialAnalysisResult(BaseModel):
    ... # Define fields relevant to social sentiment analysis

def social_agent(state:AgentState) -> dict:
    """
    Analyze social media data to extract key insights for investment decision-making.
    """
    # logger.info(f"💬 Processing social media data through Social Analyst Agent {state}")

    try:
        social_input = state['social_data']

        SOCIAL_PROMPT = f"""
        ROLE: Social Media Analyst.
        You are a financial analyst specializing in stock market predictions.
        GOAL: Analyze the provided social media summary to extract key insights relevant for investment decision-making.
        FOCUS: Identify potential risks and opportunities highlighted in social media that could impact the stock's performance.

        COMPANY : {state['ticker']}

        INPUT: {social_input}

        Based on this social media data, provide your analysis.

        """
        response=LLM.with_structured_output(SocialAnalysisResult).invoke(SOCIAL_PROMPT)
        # logger.info(f"Social Analyst Response: {response} , type  {type(response)}")

        if isinstance(response, SocialAnalysisResult):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")
        
        # Validate keys as per SocialAnalysisResult fields
        # expected_keys = {...}  # Define expected keys based on SocialAnalysisResult

        # if not expected_keys.issubset(result_dict.keys()):
        #     raise ValueError("Missing keys in the response dictionary")
        return {
            'social_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing social agent response: {e}")
        return {
            'social_analysis': {}
            }