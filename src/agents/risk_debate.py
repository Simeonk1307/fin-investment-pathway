import pathway as pw
from typing import Any


RISKY_PROMPT = """
ROLE: Risky Risk Analyst
GOAL: Advocate for high-reward opportunities and bold, asymmetric bets.
FOCUS: Downplay minor risks and emphasize the potential for massive upside. Challenge overly cautious stances. The biggest risk is missing a huge opportunity.
"""

SAFE_PROMPT = """
ROLE: Safe/Conservative Risk Analyst
GOAL: Prioritize capital preservation and minimize volatility.
FOCUS: Identify all potential downsides, black swan events, and sources of volatility. Question optimistic assumptions and advocate for stronger risk-mitigation strategies.
"""

NEUTRAL_PROMPT = """
ROLE: Neutral Risk Analyst
GOAL: Provide a balanced, data-driven perspective.
FOCUS: Objectively weigh the potential rewards presented by the Risky Analyst against the potential downsides raised by the Safe Analyst. Act as a mediator and seek a logical middle ground.
"""

PROMPT_TEMPLATE = """
{role_prompt}

Here is the investment plan to be debated:
{trader_plan}

Here is the debate so far:
{debate_history}

Your opponents' arguments are the most recent entries in the history above.
Critique or support the plan from your unique perspective and respond to your colleagues.
"""


def create_risk_debate_pipeline(
    input_stream: pw.Table,
    llm: Any,
    num_rounds: int = 1,
) -> pw.Table:
    """
    Constructs a Pathway pipeline that simulates a multi-round risk debate
    between Risky, Safe, and Neutral analysts.
    """

    def create_risk_analyst_udf(agent_name: str, role_prompt: str, llm_client: Any):
        @pw.udf
        def run_risk_analyst(trader_plan: str, debate_history: str) -> str:
            """Generates an argument for one analyst and appends it to the history."""

            prompt = PROMPT_TEMPLATE.format(
                role_prompt=role_prompt,
                trader_plan=trader_plan,
                debate_history=debate_history or "(EMPTY)",
            )

            response = llm_client.invoke(prompt).content.strip()
            new_argument = f"{agent_name}: {response}"
            new_debate_history = f"{debate_history}\n{new_argument}".strip()
            return new_debate_history
        
        return run_risk_analyst

    risk_debate_data = input_stream.with_columns(
        risk_debate_history="",
    )

    risky_udf = create_risk_analyst_udf("Risky Analyst", RISKY_PROMPT, llm)
    safe_udf = create_risk_analyst_udf("Safe Analyst", SAFE_PROMPT, llm)
    neutral_udf = create_risk_analyst_udf("Neutral Analyst", NEUTRAL_PROMPT, llm)

    for _ in range(num_rounds):
        risk_debate_data = risk_debate_data.with_columns(
            risk_debate_history=risky_udf(pw.this.trader_investment_plan, pw.this.risk_debate_history)
        )
        risk_debate_data = risk_debate_data.with_columns(
            risk_debate_history=safe_udf(pw.this.trader_investment_plan, pw.this.risk_debate_history)
        )
        risk_debate_data = risk_debate_data.with_columns(
            risk_debate_history=neutral_udf(pw.this.trader_investment_plan, pw.this.risk_debate_history)
        )

    return risk_debate_data


# # --- Example Usage Block ---
# if __name__ == "__main__":

#     # --- Mock Objects ---

#     class MockLLMResponse:
#         def __init__(self, content):
#             self.content = content

#     class MockLLM:
#         def invoke(self, prompt: str) -> MockLLMResponse:
#             first_line = next((l.strip() for l in prompt.splitlines() if l.strip()), "").lower()
#             if "risky risk analyst" in first_line:
#                 return MockLLMResponse("The 'HOLD' plan is too timid. This is a generational opportunity. The high P/E reflects market confidence. We should be buying aggressively, not sitting on the sidelines.")
#             elif "safe/conservative" in first_line:
#                 return MockLLMResponse("I disagree completely. The plan's stop-loss is too loose. A 200 P/E is a sign of a bubble, not confidence. We should be considering an outright SELL or buying put options to hedge.")
#             elif "neutral risk analyst" in first_line:
#                 return MockLLMResponse("Both points have merit. The plan's 'HOLD' recommendation correctly identifies the high volatility. However, Safe's point about the P/E is statistically significant. Perhaps the plan should be amended to 'HOLD with a tighter stop-loss at $95'.")
#             return MockLLMResponse("No response generated.")

#     # --- Sample Data (output from a previous agent) ---

#     sample_data = pd.DataFrame([{
#         'ticker': 'AI_CORP',
#         'trader_investment_plan': (
#             "**Recommendation:** HOLD.\n"
#             "**Plan:** The high volatility suggests waiting for the next earnings report before committing new capital. Set a stop-loss at $90 to protect against a sharp downturn."
#         )
#     }])

#     # --- Pipeline Execution ---

#     mock_llm = MockLLM()
#     input_table = pw.debug.table_from_pandas(sample_data)

#     print("Running Risk Debate Team Pipeline with Mock Data...\n")

#     # Construct the pipeline for a single round of debate
#     final_risk_debate_table = construct_risk_debate_pipeline(
#         input_stream=input_table,
#         llm=mock_llm,
#         num_rounds=1
#     )

#     # Run the pipeline and print the results
#     pw.debug.compute_and_print(final_risk_debate_table)