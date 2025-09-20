from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain import hub
from config import get_llm
from .tools import search_notes, quiz_me, summarize

def build_agent():
    tools = [
        Tool(
            name="Search_Notes",
            func=search_notes,
            description="Search study notes for specific information. Input should be a clear search query about the topic you need information on. Use this ONLY when you need to find specific content from the user's study materials."
        ),
        Tool(
            name="Quiz_Generator",
            func=quiz_me,
            description="Generate practice questions about a specific topic. Input should be the topic name. Use this ONLY when the user explicitly asks for quiz questions, practice questions, or testing material."
        ),
        Tool(
            name="Summarizer",
            func=summarize,
            description="Create a concise summary of provided text. Input should be the text to summarize. Use this ONLY when the user explicitly asks for a summary of specific content."
        ),
    ]
    
    llm = get_llm()
    
    # Update the prompt template for better clarity and guidance
    from langchain.prompts import PromptTemplate
    template = """
You are a helpful study assistant. You have access to the user's study notes and the following tools:

Tools:
{tools}

DECISION RULES:
- If the user asks about their notes → use Search_Notes ONCE
- If the user asks for summaries → use Summarizer ONCE
- If the user asks for quiz questions → use Quiz_Generator ONCE
- For general knowledge questions → answer directly

IMPORTANT: Always respond in EXACTLY this format. DO NOT OMIT ANY FIELDS:

Question: {input}
Thought: (your reasoning)
Action: (tool name from {tool_names} or None)
Action Input: (input for the tool or None)
Observation: (tool output or None)
Thought: I now have the information needed
Final Answer: (complete answer)

Ensure that:
1. Each tool is called at most once per question.
2. If no tool is needed, write Action: None and Action Input: None.
3. Observation should reflect the tool output if a tool was used.

Question: {input}
Thought:{agent_scratchpad}
"""
    
    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=120,
        return_intermediate_steps=False
    )
    
    return agent_executor
