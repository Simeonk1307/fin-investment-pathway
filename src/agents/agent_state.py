from typing import Literal,Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
import operator

class AgentState(TypedDict):
    """Central state that flows through all agents"""
    # Input data
    ticker: str
    market_data: dict
    news: dict

    
    # Agent outputs
    news_analysis: dict  # NEW: Structured news output
    final_analysis: dict  # Final investment decision

    bull_arguments: Sequence[str]
    bear_arguments: Sequence[str]
    
    # Metadata
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

