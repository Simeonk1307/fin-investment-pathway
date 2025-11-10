import pathway as pw
from typing import Any
import pandas as pd

# --- Agent Prompt (Constant) ---

PORTFOLIO_MANAGER_PROMPT = """
ROLE: Portfolio Manager
GOAL: Your decision is final. You are accountable for the profit and loss.
TASK:
1.  **Review the Investment Plan**: Critically assess the plan proposed by the research team (the "Trader's Plan").
2.  **Review the Risk Debate**: Consider the specific objections and concerns raised in the risk-focused debate.
3.  **Provide a Final Decision**: Issue a binding, one-word directive: **BUY**, **SELL**, or **HOLD**.
4.  **Justify the Decision**: Provide a concise, clear justification for your decision, explaining how you weighed the investment plan against the identified risks.
---
Trader's Investment Plan:
{trader_plan}

Risk Debate History:
{risk_debate}
"""

# --- Core Pipeline Construction Function ---

def construct_portfolio_manager_pipeline(
    input_stream: pw.Table,
    llm: Any,
) -> pw.Table:
    """
    Constructs a Pathway pipeline where a Portfolio Manager makes a final
    trade decision based on a proposed trader's plan and a risk debate.

    Args:
        input_stream (pw.Table):
            A Pathway table containing the inputs for the decision.
            Must include columns: 'trader_investment_plan', 'risk_debate_history'.
        llm (Any):
            An LLM client instance (e.g., from LiteLLM, Langchain) used for
            the final decision. Must have an `.invoke(prompt)` method that
            returns a response object with a `.content` attribute.

    Returns:
        pw.Table:
            The final table containing the original data plus the portfolio
            manager's final, binding trade decision.
    """

    @pw.udf
    def run_portfolio_manager(trader_plan: str, risk_debate: str) -> str:
        """
        Generates a final trade decision by invoking the Portfolio Manager LLM.
        """
        prompt = PORTFOLIO_MANAGER_PROMPT.format(
            trader_plan=trader_plan,
            risk_debate=risk_debate
        )
        response = llm.invoke(prompt).content.strip()
        return response

    # Apply the UDF to the input stream to generate the final decision
    final_decision_table = input_stream.with_columns(
        final_trade_decision=run_portfolio_manager(
            pw.this.trader_investment_plan,
            pw.this.risk_debate_history
        )
    )

    return final_decision_table


# # --- Example Usage Block ---
# # This block demonstrates how to use the function and makes the file runnable.
# if __name__ == "__main__":

#     # --- Mock Objects ---

#     class MockLLMResponse:
#         def __init__(self, content):
#             self.content = content

#     class MockLLM:
#         def invoke(self, prompt: str) -> MockLLMResponse:
#             if "Portfolio Manager" in prompt:
#                 return MockLLMResponse(
#                     "**DECISION: BUY**\n\n"
#                     "**Justification:** The growth potential outlined in the plan is compelling. "
#                     "While the valuation risk is noted, the recent technology breakthrough justifies a premium. "
#                     "We will enter a partial position now and scale in post-earnings, mitigating the risk."
#                 )
#             return MockLLMResponse("No response generated.")

#     # --- Sample Data ---
#     # This data simulates the output from previous stages of a larger pipeline.
#     sample_data = pd.DataFrame([{
#         'ticker': 'AI_CORP',
#         'trader_investment_plan': (
#             "**Summary:** Bull focuses on innovation, Bear on valuation.\n"
#             "**Recommendation:** HOLD.\n"
#             "**Plan:** The recurring disagreement on valuation vs. growth suggests high volatility. "
#             "Wait for the next earnings report. Set a tight stop-loss at $90."
#         ),
#         'risk_debate_history': (
#             "Risk Officer: The concentration of revenue from a single product is too high.\n"
#             "Compliance Officer: Agreed. A negative report on that product could be catastrophic.\n"
#             "Risk Officer: I recommend we avoid this stock until they demonstrate diversification."
#         )
#     }])

#     # --- Pipeline Execution ---

#     mock_llm = MockLLM()
#     input_table = pw.debug.table_from_pandas(sample_data)

#     print("Running Portfolio Manager Pipeline with Mock Data...\n")

#     # Construct the pipeline
#     final_decision_table = construct_portfolio_manager_pipeline(
#         input_stream=input_table,
#         llm=mock_llm
#     )
    
#     # Run the pipeline and print the results to the console
#     pw.debug.compute_and_print(final_decision_table)