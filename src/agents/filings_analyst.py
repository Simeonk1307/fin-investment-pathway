from src.agents.llm_factory import LLM
from src.agents.agent_state import AgentState
import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

class FilingsAnalysisResult(BaseModel):
    ... # Define fields relevant to filings analysis

def filings_agent(state:AgentState) -> dict:
    """
    Analyze filings data to extract key insights for investment decision-making.
    """
    # logger.info(f"📄 Processing filings data through Filings Analyst Agent {state}")

    try:
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
        
        # Validate keys as per FilingsAnalysisResult fields
        # expected_keys = {...}  # Define expected keys based on FilingsAnalysisResult

        # if not expected_keys.issubset(result_dict.keys()):
        #     raise ValueError("Missing keys in the response dictionary")
        
        return {
            'filings_analysis': result_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing filings agent response: {e}")
        return {
            "filings_analysis": {}
        }