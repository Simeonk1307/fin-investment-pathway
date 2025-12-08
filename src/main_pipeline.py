from src.input_pipeline import news_input_pipeline,social_input_pipeline, filings_input_pipeline, stock_signal_pipeline
import pathway as pw
from src.agents.agent_pipeline import  create_graph
from src.agents.llm_factory import get_llm
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
import signal
import os
import logging

load_dotenv(find_dotenv)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
def shutdown_handler(signum, frame):
    print("[Pipeline] Shutdown signal received", flush=True)
    os._exit(0)
LLM = get_llm('perplexity')

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)

def process_ticker(ticker: str,
                   news_articles: tuple[str],news_sentiment_scores: tuple[float],
                   socials_articles: tuple[str], socials_sentiment_scores: tuple[float],
                   filings_summary:tuple[str]):

    logger.info(f"[Main pipeline] Started agent graph for {ticker}")
    logger.info(f"[Main pipeline] News Sentiment scores : {news_sentiment_scores}")
    logger.info(f"[Main pipeline] Social Sentiment scores : {socials_sentiment_scores}")
    logger.info(f"[Main pipeline] No of articles : {len(news_articles)}")
    logger.info(f"[Main pipeline] No of articles : {len(socials_articles)}")
    graph = create_graph()
    state = {
        "LLM" : LLM,
        "ticker": ticker,

        "news_data":
        {
            "news_articles": news_articles,
            "news_sentiment_scores": news_sentiment_scores,
        },

        "filings_data":{
            # check filings data
            # check  filings_analyst and  rewrite prompt and define FilingsAnalysisResult
            "filings_data" : filings_summary,

        },

        "socials_data":{
            "socials_articles": socials_articles,
            "socials_sentiment_scores": socials_sentiment_scores,

        },

        "market_data":{
            # check market data
        },

        "news_analysis":{

        },

        "filings_analysis":{

        },

        "socials_analysis":{

        },

        "final_analysis":{

        },

        "messages": [HumanMessage(content=f"Analyzing {ticker}")],
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


def run_agent_pipeline(
    news_table: pw.Table,
    socials_table: pw.Table,
    filings_table: pw.Table,
    stocks_table: pw.Table,
    output_path: str = "outputs/"
):
    @pw.udf
    def get_element(analysis:dict, key:str):
        return analysis[key]
    
    # Extract ticker and prepare minimal state
    analysed_news = news_table.select(
        symbol=pw.this.symbol,
        news_articles=pw.this.news_articles,
        news_sentiment_scores=pw.this.news_sentiment_scores,
    )

    logger.info(f"[NEWS] Sentiment analysis done")

    analysed_socials = socials_table.select(
        symbol=pw.this.symbol,
        socials_articles=pw.this.socials_articles,
        socials_sentiment_scores=pw.this.socials_sentiment_scores,
    )
    analysed_filings=filings_table.select(
        symbol=pw.this.symbol,
        filing_summary=pw.this.filing_summary
    )

    logger.info(f"[SOCIALS] Sentiment analysis done")

    agents_input = analysed_news.join(analysed_socials, pw.left.symbol == pw.right.symbol)
    agents_input = agents_input.join(analysed_filings, pw.left.symbol == pw.right.symbol)
     # Process through LangGraph
    agents_output = agents_input.select(
        symbol=pw.this.symbol,
        news_sentiment_scores=pw.this.news_sentiment_scores,
        socials_sentiment_scores=pw.this.socials_sentiment_scores,
        analysis=pw.apply(
            process_ticker,
            pw.this.symbol,
            pw.this.news_articles,
            pw.this.news_sentiment_scores,
            pw.this.socials_articles,
            pw.this.socials_sentiment_scores,
            pw.this.filing_summary)
    )
    
    # Process through LangGraph
    results = agents_output.select(
        symbol=pw.this.symbol,
        news_sentiment_scores=pw.this.news_sentiment_scores,
        prediction=get_element(pw.this.analysis, "prediction"),
        confidence=get_element(pw.this.analysis, "confidence"),
        reason=get_element(pw.this.analysis, "reason"),
        strategy=get_element(pw.this.analysis, "strategy"),
    )
    if DEBUG:
        os.makedirs(output_path, exist_ok=True)
        pw.io.csv.write(results, f"{output_path}agent_pipeline_news_analysis.csv")
        pw.io.jsonlines.write(results, f"{output_path}agent_pipeline_news_analysis.jsonl")

    return results

def run_main_pipeline():
    print("[MAIN PIPELINE] Starting...", flush=True)

    news_table = news_input_pipeline()
    socials_table = social_input_pipeline()
    filings_table = filings_input_pipeline()
    # Optional: create stock signal pipeline (will be a no-op if env var not set)
    try:
        stock_table = stock_signal_pipeline()
    except Exception:
        stock_table = None

    run_agent_pipeline(
        news_table=news_table,
        socials_table=socials_table,
        filings_table=filings_table,
        stocks_table=stock_table,
        output_path="debug_output/agents/"
    )
    pw.run()

    print("[MAIN PIPELINE] Finished.", flush=True)

if __name__ == "__main__":
    run_main_pipeline()