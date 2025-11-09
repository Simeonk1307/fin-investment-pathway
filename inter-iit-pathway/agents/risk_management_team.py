risky_prompt = "You are the Risky Risk Analyst. You advocate for high-reward opportunities and bold strategies."
safe_prompt = "You are the Safe/Conservative Risk Analyst. You prioritize capital preservation and minimizing volatility."
neutral_prompt = "You are the Neutral Risk Analyst. You provide a balanced perspective, weighing both benefits and risks."

def create_risk_node(llm, role_prompt, agent_name):
    def risk_node(state):
        
        risk_state = state['risk_debate_state']
        opponents_args = []
        if agent_name != 'Risky Analyst' and risk_state['current_risky_response']: opponents_args.append(f"Risky: {risk_state['current_risky_response']}")
        if agent_name != 'Safe Analyst' and risk_state['current_safe_response']: opponents_args.append(f"Safe: {risk_state['current_safe_response']}")
        if agent_name != 'Neutral Analyst' and risk_state['current_neutral_response']: opponents_args.append(f"Neutral: {risk_state['current_neutral_response']}")
        
        prompt = f"""{role_prompt}
        Here is the trader's plan: {state['trader_investment_plan']}
        Debate history: {risk_state['history']}
        Your opponents' last arguments:\n{'\n'.join(opponents_args)}
        Critique or support the plan from your perspective."""
        
        response = llm.invoke(prompt).content
        
        new_risk_state = risk_state.copy()
        new_risk_state['history'] += f"\n{agent_name}: {response}"
        new_risk_state['current_speaker'] = agent_name
        if agent_name == 'Risky Analyst': new_risk_state['current_risky_response'] = response
        elif agent_name == 'Safe Analyst': new_risk_state['current_safe_response'] = response
        else: new_risk_state['current_neutral_response'] = response
        new_risk_state['count'] += 1

        return {"risk_debate_state": new_risk_state}
    return risk_node

risky_node = create_risk_node(quick_thinking_llm, risky_prompt, "Risky Analyst")
safe_node = create_risk_node(quick_thinking_llm, safe_prompt, "Safe Analyst")
neutral_node = create_risk_node(quick_thinking_llm, neutral_prompt, "Neutral Analyst")
