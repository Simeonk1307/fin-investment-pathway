bull_prompt = """
ROLE: Bull Analyst.
GOAL: Argue in favor of investing in a stock.
FOCUS: growth potential, competitive advantages and positive indicators from the reports.
Counter the Bear's arguments effectively.
"""

bear_prompt = """
ROLE: Bear Analyst.
GOAL: Argue against investing in a stock. 
FOCUS: risks, challenges, and negative indicators. 
Counter the bull's arguments effectively.
"""

judge_prompt = """
ROLE: Research Manager.
GOAL: Overseeing the debate between Bull and Bear Analysts and critically evaluating them before making a definitive decision
Summarize the key points, then provide a clear recommendation: Buy, Sell, or Hold. Develop a detailed investment plan for the trader, including your rationale and strategic actions.
"""

def analysis_summary(state):
    return f"""
    Market Report: {state['market_report']}
    News Report: {state['news_report']}
    Social Media Sentiment: {state['social_media_report']}
    Fundamentals Report: {state['fundamentals_report']}
    """

def past_context(memory_text, state):
    return f"""
    Conversation history: {state['research_debate_state']['history']}
    Your opponent's last argument: {state['research_debate_state']['current_response']}
    Reflections from similar past situations: {memory_text or 'No past memories found.'}
    """

def create_analyst_node(llm, memory, role_prompt, agent_name):

    def analyst_node(state):
        analysis_summary = analysis_summary(state)
        
        past_memories = memory.get_memories(analysis_summary)
        past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
        past_context = past_context(past_memory_str, state)

        prompt = f"""{role_prompt}
        Here is the current state of the analysis:
        {analysis_summary}
        {past_context}
        Based on all this information, present your argument conversationally."""
        
        response = llm.invoke(prompt).content.strip()
        argument = f"{agent_name}: {response}"
        
        debate_state = state['research_debate_state'].copy()
        debate_state['history'] += f"\n{argument}"
        debate_state['current_response'] = argument
        if agent_name == 'Bull Analyst':
            debate_state['bull_history'] += f"\n{argument}"
        else:
            debate_state['bear_history'] += f"\n{argument}"
        debate_state['count'] += 1

        return {"research_debate_state": debate_state}
    return analyst_node

def create_judge_node(llm, memory, role_prompt):

    def judge_node(state):
        prompt = f"""{role_prompt}
        
        Here is the debate history:
        {state['research_debate_state']['history']}
        """
        
        response = llm.invoke(prompt).content.strip()
        
        return {"judge_investment_plan": response}
    return judge_node


bull_analyst_node = create_analyst_node(quick_thinking_llm, bull_memory, bull_prompt, 'Bull Analyst')
bear_analyst_node = create_analyst_node(quick_thinking_llm, bear_memory, bear_prompt, 'Bear Analyst')
judge_node = create_judge_node(deep_thinking_llm, judge_memory, judge_prompt)
