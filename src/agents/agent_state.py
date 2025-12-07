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
    socials_data:dict

    LLM: any  # LLM instance
    
    # Agent outputs
    news_analysis: dict  # NEW: Structured news output
    filings_analysis: dict  # Filings analysis output

    #market_analysis: dict  # Market data analysis output
    
    socials_analysis: dict  # Social sentiment analysis output
   
    final_analysis: dict  # Final investment decision


    bull_arguments: Sequence[str]
    bear_arguments: Sequence[str]
    
    # Metadata
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

def data_ingestion_node(state: AgentState) -> AgentState:
    """
    Fetch data from Silver layer topics
    """
    if ('news_articles' not in state['news_data']) :
        raise KeyError("Missing news_articles in state")
    
    if ('news_sentiment_scores' not in state['news_data']):
        raise KeyError("Missing news sentiment scores in state")
    
    if ('socials_articles' not in state['socials_data']) :
        raise KeyError("Missing socials_articles in state")
    
    if ('socials_sentiment_scores' not in state['socials_data']):
        raise KeyError("Missing socials sentiment scores in state")
    

    return None
