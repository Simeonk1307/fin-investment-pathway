import pathway as pw
from typing import Any, Callable
import pandas as pd

# --- Agent Prompts (Constants) ---

BULL_PROMPT = """
ROLE: Bull Analyst.
GOAL: Argue in favor of investing in a stock.
FOCUS: growth potential, competitive advantages and positive indicators from the reports.
Counter the Bear's arguments effectively. If there is no opponent's argument yet, make a strong opening statement.
"""

BEAR_PROMPT = """
ROLE: Bear Analyst.
GOAL: Argue against investing in a stock. 
FOCUS: risks, challenges, and negative indicators. 
Counter the bull's arguments effectively.
"""



# --- Pipeline 1: Bull vs. Bear Debate Factory ---

def create_bull_bear_debate_builder(
    llm: Any,
    bull_memory: Any,
    bear_memory: Any,
) -> Callable[[pw.Table, int], pw.Table]:
    """
    A factory that takes dependencies and returns the actual pipeline construction function.

    Args:
        llm: The LLM client to be used for the debate.
        bull_memory: The memory system for the Bull analyst.
        bear_memory: The memory system for the Bear analyst.

    Returns:
        A function that constructs the bull-bear debate pipeline.
    """

    # This is the function that will be returned.
    # It has access to llm, bull_memory, and bear_memory from the outer scope.
    def construct_bull_bear_debate_pipeline(
        input_stream: pw.Table,
        num_rounds: int = 1,
    ) -> pw.Table:
        """
        Constructs a multi-round debate pipeline. Its dependencies are pre-configured.

        Args:
            input_stream: Table with initial analysis data.
            num_rounds: The number of debate rounds.
        """
        @pw.udf
        def format_analysis_summary(market: str, news: str, social: str, fundamentals: str) -> str:
            return f"\nMarket Report: {market}\nNews Report: {news}\nSocial Media Sentiment: {social}\nFundamentals Report: {fundamentals}"

        def create_analyst_udf(agent_name: str, role_prompt: str, memory_system: Any):
            @pw.udf
            def run_analyst(analysis_summary: str, debate_history: str) -> str:
                # Accesses memory_system and llm from the closure
                past_memories = memory_system.get_memories(analysis_summary + debate_history)
                past_memory_str = "\n".join([mem.get("recommendation", "") for mem in past_memories])
                history_lines = debate_history.strip().split('\n')
                opponent_arg = history_lines[-1] if history_lines else "None (you are making the opening statement)"
                past_context = f"Conversation history: {debate_history or '(Empty)'}\nYour opponent's last argument: {opponent_arg}\nReflections: {past_memory_str or 'None'}"
                prompt = f"{role_prompt}\nHere is the analysis:\n{analysis_summary}\n\n{past_context}\n\nPresent your argument:"
                response = llm.invoke(prompt).content.strip()
                new_argument = f"{agent_name}: {response}"
                return f"{debate_history}\n{new_argument}".strip()
            return run_analyst

        analyzed_data = input_stream.with_columns(
            analysis_summary=format_analysis_summary(pw.this.market_report, pw.this.news_report, pw.this.social_media_report, pw.this.fundamentals_report),
            debate_history="",
        )

        # UDFs are created using the dependencies from the factory scope
        bull_udf = create_analyst_udf("Bull Analyst", BULL_PROMPT, bull_memory)
        bear_udf = create_analyst_udf("Bear Analyst", BEAR_PROMPT, bear_memory)

        debated_data = analyzed_data
        for _ in range(num_rounds):
            debated_data = debated_data.with_columns(debate_history=bull_udf(pw.this.analysis_summary, pw.this.debate_history))
            debated_data = debated_data.with_columns(debate_history=bear_udf(pw.this.analysis_summary, pw.this.debate_history))

        return debated_data

    return construct_bull_bear_debate_pipeline




# # --- Example Usage Block: Chaining the Pipelines ---
# if __name__ == "__main__":
    
#     # --- Mock Objects and Config ---
#     config = {"num_rounds": 2}
#     class MockLLMResponse:
#         def __init__(self, content): self.content = content
#     class MockLLM:
#         def invoke(self, prompt: str) -> MockLLMResponse:
#             first_line = next((l.strip() for l in prompt.splitlines() if l.strip()), "").lower()
#             if "role: bull analyst" in first_line: return MockLLMResponse("Round 1: Growth is unstoppable.") if "Bull Analyst" not in prompt else MockLLMResponse("Round 2: Innovation wins.")
#             if "role: bear analyst" in first_line: return MockLLMResponse("Round 1: Valuation is absurd.") if "Bear Analyst" not in prompt else MockLLMResponse("Round 2: P/E is too high.")
#             if "role: research manager" in first_line: return MockLLMResponse("**Recommendation:** HOLD.")
#             return MockLLMResponse("No response.")
#     class MockMemory:
#         def get_memories(self, query: str) -> list: return []

#     # --- SETUP PHASE: CONFIGURE DEPENDENCIES ---
#     quick_thinking_llm = MockLLM()
#     deep_thinking_llm = MockLLM()
#     bull_memory = MockMemory()
#     bear_memory = MockMemory()

#     # 1. Call the factories to get your pre-configured pipeline functions.
#     #    This happens once during initialization.
#     debate_pipeline_fn = create_bull_bear_debate_builder(
#         llm=quick_thinking_llm,
#         bull_memory=bull_memory,
#         bear_memory=bear_memory
#     )
#     manager_pipeline_fn = create_research_manager_builder(
#         llm=deep_thinking_llm
#     )
    
#     # --- DATA and EXECUTION PHASE ---
#     sample_data = pd.DataFrame([{'ticker': 'AI_CORP', 'market_report': 'Tech bullish', 'news_report': 'Breakthrough', 'social_media_report': 'Positive', 'fundamentals_report': 'High P/E'}])
#     input_table = pw.debug.table_from_pandas(sample_data)

#     print(f"Running Research Team Pipeline with {config['num_rounds']} debate rounds...\n")
    
#     # 2. Call the generated functions with clean, data-focused signatures.
#     #    Notice no memory or LLM objects are passed here.
#     debate_results_table = debate_pipeline_fn(
#         input_stream=input_table,
#         num_rounds=config['num_rounds']
#     )
    
#     final_analysis_table = manager_pipeline_fn(
#         input_stream=debate_results_table
#     )
    
#     # --- Final Output ---
#     output_view = final_analysis_table.select(pw.this.ticker, pw.this.debate_history, pw.this.judge_investment_plan)
#     print("\n--- Final Output ---")
#     pw.debug.compute_and_print(output_view)