import pathway as pw
from typing import Any, Callable
import pandas as pd

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

ANALYSIS_SUMMARY = """
Market Report: {market_report}
News Report: {news_report}
Social Media Sentiment: {social_media_report}
Fundamentals Report: {fundamentals_report}
"""

PROMPT_TEMPLATE = """
{role_prompt}

Here is the analysis:
{analysis_summary}

Conversation history:
{debate_history}

Your opponent's last argument: 
{opponent_arg}

Reflections: 
{past_memory_str}

Present your argument:
"""


def create_bull_bear_debate_pipeline(
    input_stream: pw.Table,
    llm: Any,
    bull_memory: Any,
    bear_memory: Any,
    num_rounds: int = 1,
) -> pw.Table:
    """
    Constructs a multi-round debate pipeline. 
    Its dependencies are pre-configured.
    """

    def create_analyst_udf(agent_name: str, role_prompt: str, memory_system: Any):
        @pw.udf
        def run_analyst(analysis_summary: str, debate_history: str) -> str:
            past_memories = memory_system.get_memories(analysis_summary)
            past_memory_str = "\n".join([mem.get("recommendation", "") for mem in past_memories])

            history_lines = debate_history.strip().split('\n')
            opponent_arg = history_lines[-1] if history_lines else None

            prompt = PROMPT_TEMPLATE.format(
                role_prompt=role_prompt,
                analysis_summary=analysis_summary,
                debate_history=debate_history or "(EMPTY)",
                opponent_arg=opponent_arg or "(NONE)",
                past_memory_str=past_memory_str or "(NONE)",
            )

            response = llm.invoke(prompt).content.strip()
            new_argument = f"{agent_name}: {response}"
            new_debate_history = f"{debate_history}\n{new_argument}".strip()
            return new_debate_history
        return run_analyst

    bull_bear_table = input_stream.with_columns(
        analysis_summary=ANALYSIS_SUMMARY.format(
            market_report=pw.this.market_report,
            news_report=pw.this.news_report,
            social_media_report=pw.this.social_media_report,
            fundamentals_report=pw.this.fundamentals_report,
        ),
        debate_history="",
    )
    
    bull_udf = create_analyst_udf("Bull Analyst", BULL_PROMPT, bull_memory)
    bear_udf = create_analyst_udf("Bear Analyst", BEAR_PROMPT, bear_memory)

    debated_data = bull_bear_table
    for _ in range(num_rounds):
        debated_data = debated_data.with_columns(debate_history=bull_udf(pw.this.analysis_summary, pw.this.debate_history))
        debated_data = debated_data.with_columns(debate_history=bear_udf(pw.this.analysis_summary, pw.this.debate_history))

    return debated_data





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