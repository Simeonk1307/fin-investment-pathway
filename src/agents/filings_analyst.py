from src.agents.agent_state import AgentState
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)


class FilingsAnalysisResult(BaseModel):
    prediction: str = Field(..., description="Stock price movement prediction: UP, DOWN, or NEUTRAL")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    reason: str = Field(..., description="Brief reason for the prediction in one line. It should be clear and specific")
def filings_agent(state:AgentState) -> dict:
    """
    Analyze filings data to extract key insights for investment decision-making.
    """
    # logger.info(f"📄 Processing filings data through Filings Analyst Agent {state}")

    try:
        LLM = state['LLM']
        filings_input = state['filings_data']

        FILINGS_PROMPT = f"""
        ROLE: Filings Analyst.
        You are a financial analyst specializing in stock market predictions.
        GOAL: Analyze the provided filings summary to extract key insights relevant for investment decision-making.
        FOCUS: Identify potential risks and opportunities highlighted in the filings that could impact the stock's performance.

        COMPANY : {state['ticker']}

        INPUT: {filings_input}

        Based on this filings data, provide your analysis.

        """
        response=LLM.with_structured_output(FilingsAnalysisResult).invoke(FILINGS_PROMPT)
        # logger.info(f"Filings Analyst Response: {response} , type  {type(response)}")

        if isinstance(response, FilingsAnalysisResult):
            result_dict = response.model_dump()
        else:
            raise ValueError("Response content is neither dict nor str")

        
        # Validate keys
        expected_keys = {"prediction", "confidence", "reason"}

        if not expected_keys.issubset(result_dict.keys()):
            raise ValueError("Missing keys in the response dictionary")
        
        logger.info(f"[AGENT PIPELINE] Filings analysis received : {result_dict}")
        
        return {
            'filings_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing filings agent response: {e}")
        return {
            "filings_analysis": {}
        }