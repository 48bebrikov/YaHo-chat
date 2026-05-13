import os
import asyncio
import logging
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import GEMINI_API_KEY, GEMINI_MODEL_ID
from ai.tools import search_internet, search_youtube, search_saved_news, browse_url
from ai.tools_sota import execute_python_code, add_user_reminder

logger = logging.getLogger(__name__)

# --- State Definition ---
class AgentState(TypedDict):
    # 'add_messages' ensures that new messages append to the list rather than overwrite
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str

# --- Tools Conversion ---
# We wrap our existing tools into LangChain @tool

@tool
def google_search(query: str) -> str:
    """Searches the internet for information."""
    return search_internet(query=query)

@tool
def youtube_search(query: str) -> str:
    """Searches YouTube for videos."""
    return search_youtube(query=query)

@tool
def news_search(query: str) -> str:
    """Searches saved channel news."""
    return search_saved_news(query=query)

@tool
def get_webpage(url: str) -> str:
    """Fetches the text content of a URL."""
    return browse_url(url=url)

@tool
def run_python(code: str) -> str:
    """
    Executes Python code. Useful for math, calculations, or data processing.
    Pass the raw python code as the argument.
    """
    return execute_python_code(code)

@tool
def set_reminder(user_id: str, text: str, delay_minutes: int) -> str:
    """
    Sets a reminder for the user.
    `user_id`: The ID of the user (you can find this in your system prompt).
    `text`: What to remind the user about.
    `delay_minutes`: How many minutes from now to send the reminder.
    """
    return add_user_reminder(user_id, text, delay_minutes)

tools = [google_search, youtube_search, news_search, get_webpage, run_python, set_reminder]
tool_node = ToolNode(tools)

# --- LLM Node ---
def get_llm():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    # Bind the tools to the LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_ID,
        api_key=GEMINI_API_KEY,
        temperature=0.7,
    )
    return llm.bind_tools(tools)

def call_model(state: AgentState):
    """Invokes the agent model to generate a response based on the current state."""
    messages = state["messages"]
    llm = get_llm()
    
    try:
        response = llm.invoke(messages)
        # We return a dict, because this is added to the state
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in LLM call: {e}")
        # Return a fallback message to avoid hanging
        fallback = AIMessage(content="Прости, что-то с интернетом, не могу ответить.")
        return {"messages": [fallback]}

# --- Conditional Edge ---
def should_continue(state: AgentState) -> Literal["tools", END]:
    """Determines if the agent should use a tool or finish."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then we route to the "tools" node
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    # Otherwise, we stop and reply to the user
    return END

# --- Build Graph ---
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set the entrypoint
    workflow.add_edge(START, "agent")
    
    # Decide whether to finish or use tools
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    
    # After tools run, go back to the agent to evaluate the tool output
    workflow.add_edge("tools", "agent")
    
    # Compile
    # We can add memory here if we want persistence across turns, but for now
    # our `gemini_engine.py` passes the whole history + RAG, so it's stateless here.
    return workflow.compile()

# Singleton graph instance
react_agent = build_agent_graph()

async def run_react_agent(system_prompt: str, user_messages: list, user_id: str) -> str:
    """
    Entry point to run the LangGraph ReAct agent.
    Takes the system prompt and conversation history, runs the graph, and returns the final string.
    """
    # Inject user_id into the system prompt so the LLM knows it for tools
    sys_prompt_with_id = system_prompt + f"\n\n[SYSTEM] Current user_id: {user_id}"
    messages = [SystemMessage(content=sys_prompt_with_id)]
    
    # user_messages is expected to be a list of dicts like: {"role": "user"|"bot", "content": "text"} 
    # Or just raw strings if we concat them.
    # In gemini_engine, it passes a huge string of history. Let's adapt that.
    
    for msg in user_messages:
        if isinstance(msg, str):
            messages.append(HumanMessage(content=msg))
        elif isinstance(msg, BaseMessage):
            messages.append(msg)
            
    # Initial state
    state = {
        "messages": messages,
        "user_id": user_id
    }
    
    # Astream allows us to get intermediate steps if we want
    # but we just want the final result
    try:
        # We use asyncio.to_thread because the graph execution might be synchronous in some parts
        # though LangGraph supports async out of the box with `ainvoke`.
        final_state = await react_agent.ainvoke(state)
        
        last_message = final_state["messages"][-1]
        return last_message.content
        
    except Exception as e:
        logger.error(f"Failed to run LangGraph agent: {e}")
        return "Блин, чет интернет отвалился..."
