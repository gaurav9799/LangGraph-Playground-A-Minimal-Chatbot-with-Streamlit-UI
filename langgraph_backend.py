"""
A minimal LangGraph-based chatbot implementation using Azure OpenAI.

This module demonstrates how to:
- Defanine a graph-based conversational workflow using LgGraph
- Manage chat state via a TypedDict and message aggregation
- Integrate Azure OpenAI as the LLM backend
- Use an in-memory checkpointer for lightweight state persistence

The implementation is intentionally simple and designed for learning
and experimentation rather than production use.
"""

import os
import requests
from typing import TypedDict, Annotated
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv(key='AZURE_OPENAI_API_KEY')
endpoint=os.getenv(key='AZURE_OPENAI_ENDPOINT')
deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
api_version=os.getenv('AZURE_OPENAI_API_VERSION')

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment=deployment_name,
    api_version=api_version
)
search_tool=DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools=[calculator, search_tool, get_stock_price]
llm_with_tools=llm.bind_tools(tools)
class ChatState(TypedDict):
    """
    Represents the state of the chatbot within the LangGraph workflow.

    Attributes:
        messages (list[BaseMessage]):
            A list of chat messages exchanged so far.
            The `add_messages` annotation enables LangGraph to
            automatically append new messages to the existing state
            as the graph executes.
    """
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """
     Core chat node responsible for generating the assistant response.

    This function:
    - Receives the current conversation state
    - Extracts the accumulated chat messages
    - Invokes the Azure OpenAI chat model with the full message history
    - Returns the model's response in a format compatible with
      LangGraph's message aggregation mechanism

    Args:
        state (ChatState): The current state of the conversation graph.

    Returns:
        dict:
            A dictionary containing the newly generated assistant message,
            which LangGraph merges into the existing message list.
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node=ToolNode(tools)

conn=sqlite3.connect(
    database='chatbot.db',
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
