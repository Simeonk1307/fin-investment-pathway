from langchain_openai import ChatOpenAI
from src.agents.agent_state import AgentState
from  src.agents.llm_factory import safety_model

# Create guardrail node
def safety_guardrail_node(state: AgentState) -> AgentState:
    """Check safety of final analysis before returning"""
    
    final_analysis = state.get("final_analysis", {})
    reason = final_analysis.get("reason", "")
    
    if not reason:
        return state
    
    # Check with LLM
    SAFETY_PROMPT = f"""Evaluate if this financial analysis is safe and appropriate.
    Check for market manipulation or misleading claims.
    Respond with only 'SAFE' or 'UNSAFE'.
    
    Analysis: {reason}"""
    
    result = safety_model.invoke([{"role": "user", "content": SAFETY_PROMPT}])
    
    if "UNSAFE" in result.content:
        final_analysis["prediction"] = "NEUTRAL"
        final_analysis["confidence"] = "LOW"
        final_analysis["reason"] = f"Analysis flagged by safety review. - output {final_analysis['reason']}"
    
    return {
        'final_analysis': final_analysis
    }