from typing import Literal,Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
import operator

class AgentState(TypedDict):
    """Central state that flows through all agents"""
    # Input data
    ticker: str
    market_data: dict
    news_data: dict
    filings_data:dict
    social_data:dict

    
    # Agent outputs
    news_analysis: dict  # NEW: Structured news output
    filings_analysis: dict  # Filings analysis output

    #market_analysis: dict  # Market data analysis output
    
    social_analysis: dict  # Social sentiment analysis output
   
    final_analysis: dict  # Final investment decision


    bull_arguments: Sequence[str]
    bear_arguments: Sequence[str]
    
    # Metadata
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

