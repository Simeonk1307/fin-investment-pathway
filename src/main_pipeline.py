from src.input_pipeline import news_input_pipeline,social_input_pipeline, stock_signal_pipeline
import pathway as pw
from src.agents.agent_pipeline import run_agent_pipeline



def run_main_pipeline():
    print("[MAIN PIPELINE] Starting...", flush=True)

    news_table = news_input_pipeline()
    social_table = social_input_pipeline()
    # Optional: create stock signal pipeline (will be a no-op if env var not set)
    try:
        stock_table = stock_signal_pipeline()
    except Exception:
        stock_table = None

    run_agent_pipeline(
        news_table=news_table,
        # market_table=market_table,
        # fundamentals_table=fundamentals_table,
        # sentiment_table=sentiment_table,
        output_path="debug_output/agents/"
    )
    pw.run()

    print("[MAIN PIPELINE] Finished.", flush=True)

if __name__ == "__main__":
    run_main_pipeline()