from langchain.agents import create_react_agent, AgentExecutor, Tool
from config import get_llm
from .tools import search_notes, quiz_me, summarize

def build_agent():
    tools = [
        Tool(
            name="Search_Notes",
            func=search_notes,
            description="Search the user's course files (PDF, Markdown, notebooks). Input should be a clear search query. Each result starts with its source in square brackets; cite those sources in your answer. Use this ONLY when you need specific content from the user's study materials."
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

{tools}

DECISION RULES:
- If the user asks about their notes → use Search_Notes ONCE
- If the user asks for summaries → use Summarizer ONCE  
- If the user asks for quiz questions → use Quiz_Generator ONCE
- For general knowledge questions → answer directly without using any tools

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Ensure that:
1. Each tool is called at most once per question.
2. If no tool is needed, skip the Action/Action Input/Observation and go directly to Final Answer.
3. Observation should reflect the tool output if a tool was used.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
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
