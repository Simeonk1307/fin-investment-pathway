import pathway as pw
from typing import Any

TRADER_PROMPT = """
ROLE: A decisive and action-oriented Trading Agent.
GOAL: Translate a high-level investment plan into a concrete, actionable trading proposal.

TASK:
1.  Review the provided investment plan from the research team.
2.  Formulate a concise trading strategy based on the plan's recommendations (e.g., entry points, position sizing, stop-loss).
3.  Conclude your entire response with a single, unambiguous line in the format:
    FINAL TRANSACTION PROPOSAL: **BUY**, **SELL**, or **HOLD**.
---
Proposed Investment Plan:
{investment_plan}
"""

def create_trader_pipeline(
    input_stream: pw.Table,
    llm: Any,
) -> pw.Table:
    """
    Constructs a Pathway pipeline where a Trader Agent creates a specific
    trading plan and extracts a final BUY/SELL/HOLD proposal.
    """

    @pw.udf
    def run_trader_agent(plan: str) -> str:
        prompt = TRADER_PROMPT.format(investment_plan=plan)
        response = llm.invoke(prompt).content.strip()
        return response

    @pw.udf
    def extract_final_proposal(trader_plan: str) -> str:
        marker = "FINAL TRANSACTION PROPOSAL:"
        try:
            proposal_line = next(line for line in trader_plan.splitlines() if marker in line)
            if "**BUY**" in proposal_line:
                return "BUY"
            if "**SELL**" in proposal_line:
                return "SELL"
            if "**HOLD**" in proposal_line:
                return "HOLD"
        except StopIteration:
            pass
        return "UNKNOWN"

    trader_table = input_stream.with_columns(
        trader_investment_plan=run_trader_agent(pw.this.research_team_plan),
        final_proposal=extract_final_proposal(pw.this.trader_investment_plan)
    )

    return trader_table


# # --- Example Usage Block ---
# if __name__ == "__main__":

#     # --- Mock Objects ---

#     class MockLLMResponse:
#         def __init__(self, content):
#             self.content = content

#     class MockLLM:
#         def invoke(self, prompt: str) -> MockLLMResponse:
#             response_text = (
#                 "Based on the 'HOLD' recommendation and high volatility, my plan is to avoid new positions.\n"
#                 "I will set an alert at the $90 stop-loss and another at $120 to re-evaluate.\n"
#                 "No new capital will be deployed at this time.\n"
#                 "FINAL TRANSACTION PROPOSAL: **HOLD**"
#             )
#             return MockLLMResponse(response_text)

#     # --- Sample Data (output from the research team agent) ---

#     sample_data = pd.DataFrame([{
#         'ticker': 'AI_CORP',
#         'research_team_plan': (
#             "**Recommendation:** HOLD.\n"
#             "**Plan:** The high volatility suggests waiting for the next earnings report. "
#             "Set a stop-loss at $90 to protect against a sharp downturn."
#         )
#     }])

#     # --- Pipeline Execution ---

#     mock_llm = MockLLM()
#     input_table = pw.debug.table_from_pandas(sample_data)

#     print("Running Trader Agent Pipeline with Mock Data...\n")

#     # Construct the pipeline
#     final_trader_table = construct_trader_pipeline(
#         input_stream=input_table,
#         llm=mock_llm
#     )
    
#     # Run the pipeline and print the results
#     pw.debug.compute_and_print(final_trader_table)