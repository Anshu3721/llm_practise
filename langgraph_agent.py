from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Define tools
@tool
def get_refund_policy() -> str:
    """Returns the refund policy of the company."""
    return "Customers can request a refund within 30 days of purchase. Refunds are processed in 5-7 business days."

@tool
def get_shipping_info() -> str:
    """Returns shipping information."""
    return "Standard shipping takes 5-7 days. Express shipping takes 1-2 days and costs $15 extra."

@tool
def escalate_to_human(issue: str) -> str:
    """Escalates the customer issue to a human agent when the bot cannot resolve it."""
    return f"Issue '{issue}' has been escalated to a human agent. You will be contacted within 24 hours."

tools = [get_refund_policy, get_shipping_info, escalate_to_human]
llm_with_tools = llm.bind_tools(tools)

# Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Define Nodes
def llm_node(state: AgentState):
    system = SystemMessage(content="""You are a helpful customer support assistant.
    You have tools for refund policy, shipping info, and escalation.
    If question is unrelated to these, answer directly from your knowledge.
    Do NOT use tools not provided to you.""")
    
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

# Tool node - LangGraph handles tool execution automatically
tool_node = ToolNode(tools)

# Conditional edge function - decides what happens after LLM node
def should_use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "use_tool"
    return "end"

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

# Add edges
graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_use_tool,
    {
        "use_tool": "tools",
        "end": END
    }
)
graph.add_edge("tools", "llm")

# Compile the graph
app = graph.compile()

# Run the agent
def run_agent(question):
    print(f"\nUser: {question}")
    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })
    last_message = result["messages"][-1]
    output = last_message.content if last_message.content else result["messages"][-2].content
    print(f"Agent: {output}")

# Test
run_agent("What is your refund policy?")
run_agent("How long does shipping take?")
run_agent("My order is completely wrong and I am very frustrated!")
run_agent("What is the capital of France?")