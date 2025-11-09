trader_prompt = """
You are a trading agent. 
Based on the provided investment plan, create a concise trading proposal."""


def create_trader_node(llm, memory, role_prompt, agent_name):
    def trader_node(state):
        prompt = f"""{role_prompt} 
        Your response must end with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.
        
        Proposed Investment Plan: {state['investment_plan']}"""
        response = llm.invoke(prompt).content.strip()
        
        return {"trader_investment_plan": response, "sender": agent_name}
    return trader_node


trader_node = create_trader_node(quick_thinking_llm, trader_memory, trader_prompt)