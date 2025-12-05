from asyncio.log import logger
import pathway as pw
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

from src.agents.news_analyst import news_agent
from src.agents.final_agent import final_agent
from src.agents.llm_factory import get_llm
from src.agents.agent_state import AgentState
from src.agents.guard_rail import safety_guardrail_node
from src.agents.filings_analyst import filings_agent
from src.agents.social_analyst import social_agent

from datetime import datetime
import psutil
from src.agents.finbert import FinBertSentimentAnalyzer
# from src.schemas.silver_schemas import FinnHubNewsSchema
 
#if ctrl+c is pressed, stop the program
import signal
import sys
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
#even in pathway run, we can catch ctrl+c
# ============================================================================

class FinnHubNewsSchema(pw.Schema):
    id: int
    # news_id: int  # The 137618953 field
    headline: str
    description: str
    url: str
    source: str
    published_at: str
    category: str
    company: str

# class FinnHubNewsSchema(pw.Schema):
#     news_id: int
#     symbol: str
#     timestamp: int
#     source: str
#     category: str
#     title: str
#     content: str
#     url: str
#     image_url: str
        
# finnhub_news_mapping = {
#     "news_id": "id",
#     "symbol": "related",
#     "timestamp": "datetime",
#     "title": "headline",
#     "content": "summary",
#     "image_url": "image",
# }


# ============================================================================
# AGENT NODES
# ============================================================================
def data_ingestion_node(state: AgentState) -> AgentState:
    """
    Fetch latest data from Silver layer topics
    
    TODO: In future, this will receive multiple Pathway tables:
        - news_table (current)
        - market_data_table
        - fundamentals_table
        - social_sentiment_table
        - filings_table
    
    For now, only processes news (headline + summary already in state)
    """
    
    # News data already in state (headline, summary)
    # Future: Add market data, fundamentals, etc.
    # state["messages"] = [HumanMessage(content=f"Analyzing {state['ticker']}")]
    # state["debate_rounds"] = 0
    # state["news"]={
    #     "ticker": state.get("ticker",""),
    #     "headline": state.get("headline",""),
    #     "description": state.get("description",""),
    # }
    
    "Need to verify all inputs and outputs are present (keys) in state"
    return None


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================
def create_graph() -> StateGraph:
    """
    Create LangGraph workflow
    
    Current: data_ingestion → news_analysis → END
    Future: Add more agents (market_analyst, fundamental_analyst, etc.)
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("news_analysis", news_agent)
    workflow.add_node("filings_analysis", filings_agent)
    workflow.add_node("social_analysis", social_agent)
    workflow.add_node("final_analysis", final_agent)
    workflow.add_node("safety_guardrail", safety_guardrail_node)

    workflow.set_entry_point("data_ingestion")

    workflow.add_edge("data_ingestion", "news_analysis")
    workflow.add_edge("data_ingestion", "filings_analysis")
    workflow.add_edge("data_ingestion", "social_analysis")

    workflow.add_edge("news_analysis", "final_analysis")
    workflow.add_edge("filings_analysis", "final_analysis")
    workflow.add_edge("social_analysis", "final_analysis")

    workflow.add_edge("final_analysis", "safety_guardrail")
    workflow.add_edge("safety_guardrail", END)

    return workflow.compile()


def process_ticker(ticker: str,news_articles: tuple[str],news_sentiment_scores: tuple[float]):

    graph = create_graph()
    state = {
        "ticker": ticker,

        "news_data":
        {
            "news_articles": news_articles,
            "news_sentiment_scores": news_sentiment_scores,
        },

        "filings_data":{

        },

        "social_data":{

        },

        "market_data":{

        },

        "news_analysis":{

        },

        "filings_analysis":{

        },

        "social_analysis":{

        },

        "final_analysis":{

        },

        "messages": [HumanMessage(content=f"Analyzing {ticker}")],
        "debate_rounds": 0,
    }
    # logger.info(state)  
    # Run through graph
    result = graph.invoke(state)
    # logger.info(f"DEBUG - process_news_row result: {result.keys()}, type: {type(result)},analysis : {result.get('final_analysis',{})}")
    return result.get("final_analysis", {
        "prediction": "NEUTRAL",
        "confidence": "LOW",
        "reason": "Unable to analyze the news due to processing error in process_news_row."
    })

def run_pipeline(
    news_table: pw.Table,
    # market_table: pw.Table = None, 
    # fundamentals_table: pw.Table = None, 
    # sentiment_table: pw.Table = None, 
    output_path: str = "outputs/"
):
    @pw.udf
    def get_element(analysis:dict, key:str):
        return analysis[key]
    
    @pw.udf
    def merge(headline, description)->str:
        return f"Headline: {headline}\nDescription: {description}"

    @pw.udf
    def get_sentiment(headline:str="", description:str = "")->tuple[float, float, float]:
        
        text = f"Headline: {headline}\nDescription: {description}"
        return finbert_analyzer.analyze_sentiment(text)

    graph = create_graph()
    finbert_analyzer = FinBertSentimentAnalyzer()

    #if running in test mode
    news_analysis_table = news_table.groupby(pw.this.company).reduce(
        symbol=pw.this.company,
        articles=pw.reducers.tuple(merge(pw.this.headline, pw.this.description)),
        sentimental_scores=pw.reducers.tuple(get_sentiment(pw.this.headline, pw.this.description)),

    )
    
    # Extract ticker and prepare minimal state
    # news_analysis_table = news_table.groupby(pw.this.symbol).reduce(
    #     symbol=pw.this.symbol,
    #     articles = pw.reducers.tuple(merge(pw.this.title, pw.this.content)),
    #     sentimental_scores = pw.reducers.tuple(get_sentiment(pw.this.title, pw.this.content)),
    # )

    agents_input = news_analysis_table.select(
        symbol=pw.this.symbol,
        news_articles=pw.this.articles,
        news_sentiment_scores=pw.this.sentimental_scores,
    )
    agents_output = agents_input.select(
        symbol=pw.this.symbol,
        news_sentiment_scores=pw.this.news_sentiment_scores,
        analysis=pw.apply(
            process_ticker,
            pw.this.symbol,
            pw.this.news_articles,
            pw.this.news_sentiment_scores)
    )
    
    # Process through LangGraph
    results = agents_output.select(
        symbol=pw.this.symbol,
        news_sentiment_scores=pw.this.news_sentiment_scores,
        prediction=get_element(pw.this.analysis, "prediction"),
        confidence=get_element(pw.this.analysis, "confidence"),
        reason=get_element(pw.this.analysis, "reason"),
    )

    return results



if __name__ == "__main__":
    print("🧪 TEST MODE - CSV Streaming")
    print("=" * 60)
    
    csv_path = "outputs/finnhub_news.csv"
    mode = "static"
    output_path = "outputs/"
    
    # Read CSV as Pathway table (simulates Redpanda stream)
    news_table = pw.io.csv.read(
        csv_path,
        schema=FinnHubNewsSchema,
        mode=mode,
        autocommit_duration_ms=1000
    )

    results = run_pipeline(
        news_table=news_table,
        output_path=output_path
    )
    
    # Output to files
    pw.io.csv.write(results, f"{output_path}news_analysis.csv")
    # pw.io.jsonlines.write(results, f"{output_path}news_analysis.jsonl")
    
    print("🚀 Pipeline Running")
    print("=" * 60)
    print(f"📊 Input:  news_table (CSV stream)")
    print(f"📁 Output: {output_path}news_analysis.csv")
    print(f"🔍 State:  headline + summary only")
    print("=" * 60)
    
    pw.run()

