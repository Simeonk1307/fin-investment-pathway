portfolio_manager_prompt = """You are the Portfolio Manager. Based on the trader's investment plan and the risk debate, provide a final trade decision."""


def create_risk_manager(llm, memory):
    def risk_manager_node(state):
        prompt = f""" 
        As the Portfolio Manager, your decision is final. 
        Review the trader's plan and the risk debate.
        Provide a final, binding decision: Buy, Sell, or Hold, and a brief justification.
        
        Trader's Plan: {state['trader_investment_plan']}
        Risk Debate: {state['risk_debate_state']['history']}"""
        response = llm.invoke(prompt).content
        
        return {"final_trade_decision": response}
    return risk_manager_node


risk_manager_node = create_risk_manager(deep_thinking_llm, risk_manager_memory)