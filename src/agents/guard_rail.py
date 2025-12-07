from langchain_openai import ChatOpenAI
from src.agents.agent_state import AgentState
from  src.agents.llm_factory import safety_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)


# Create guardrail node
def safety_guardrail_node(state: AgentState) -> AgentState:
    """Check safety of final analysis before returning"""
    
    final_analysis = state.get("final_analysis", {})
    reason = final_analysis.get("reason", "")
    strategy = final_analysis.get("strategy", "")
    
    if not reason:
        return state
    
    # Check with LLM
    SAFETY_PROMPT = f"""Evaluate if this financial analysis is safe and appropriate.
    Check for market manipulation or misleading claims.
    Respond with only 'SAFE' or 'UNSAFE'.
    
    Analysis: {reason} . strategy : {strategy}
    """
    
    result = safety_model.invoke([{"role": "user", "content": SAFETY_PROMPT}])
    
    if "UNSAFE" in result.content:
        logger.info(f"[AGENT PIPELINE] GUARDRAIL - Annalysis flagged bu safety review")
        final_analysis["prediction"] = "NEUTRAL"
        final_analysis["confidence"] = "LOW"
        final_analysis["reason"] = f"Analysis flagged by safety review. - output {final_analysis['reason']}"
        final_analysis["strategy"] = "Unable to determine strategy. Best to HOLD or SELL."
    return {
        'final_analysis': final_analysis
    }