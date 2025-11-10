import pathway as pw
from typing import Any, Callable
import pandas as pd


JUDGE_PROMPT = """
ROLE: Research Manager.
GOAL: Overseeing the debate between Bull and Bear Analysts and critically evaluating them before making a definitive decision.
TASK: 
1.  **Summarize Key Points**: Briefly summarize the strongest argument from the Bull and the strongest argument from the Bear across the entire debate.
2.  **Provide a Clear Recommendation**: State your final verdict clearly: BUY, SELL, or HOLD.
3.  **Develop a Detailed Investment Plan**: Provide a detailed investment plan for a trader. Include your rationale, specific price targets (if applicable), stop-loss levels, and strategic actions based on the debate and your final recommendation.
"""


def construct_research_manager_pipeline(
    input_stream: pw.Table,
    llm: Any,
) -> pw.Table:
    """Constructs the judgment pipeline using its pre-configured LLM."""
    @pw.udf
    def run_judge(debate_history: str) -> str:
        prompt = f"{JUDGE_PROMPT}\n\nHere is the full debate history:\n{debate_history}"
        response = llm.invoke(prompt).content.strip() # llm comes from the factory scope
        return response
    
    return input_stream.with_columns(
        judge_investment_plan=run_judge(pw.this.debate_history)
    )
