from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Define tools using @tool decorator
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

# All tools in a dictionary for easy lookup
tools_map = {
    "get_refund_policy": get_refund_policy,
    "get_shipping_info": get_shipping_info,
    "escalate_to_human": escalate_to_human
}

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(list(tools_map.values()))

def run_agent(user_question):
    print(f"\nUser question: {user_question}")

    # Step 1 - send question to LLM
    system_message = SystemMessage(content="""You are a customer support assistant. 
                            You have access to tools for refund policy, shipping info, and escalation.
                            If the question is not related to these topics, answer directly from your own knowledge.
                            Do NOT use any tools that are not provided to you.""")

    message = [system_message, HumanMessage(content=user_question)]
    response = llm_with_tools.invoke(message)
    message.append(response)

    # print(f"message: {message} \nresponse: {response}")
    # print("----------------")


    # step 2 - check if LLM wants to use a tool
    if response.tool_calls:
        print(f"\nLLM decided to use tool: {response.tool_calls[0]['name']}")

        # Step 3 - Execute each tool the LLM requested
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']   # tool name- tool_calls=[{'name': 'get_refund_policy', 'args': {}, 'id': 'gxfnd9dmc', 'type': 'tool_call'}]
            tool_args = tool_call["args"]

            # Execute the tool
            tool_result = tools_map[tool_name].invoke(tool_args)
            print(f"\nTool result: {tool_result}")

            # Add tool result to messages FIRST
            message.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))

            # Step 4 - Now ask LLM to respond based on tool result
            final_response = llm_with_tools.invoke(message)
            if final_response.content:
                print(f"Final Answer: {final_response.content}")
            else:
                # LLM returned empty - extract answer directly from tool result
                tool_results = [msg.content for msg in message if isinstance(msg, ToolMessage)]
                print(f"Final Answer: {tool_results[-1]}")
    else:
        # LLM answer directly without using any tool
        print(f"\nLLM response without tool: {response.content}")

 # Test with different questions
run_agent("What is your refund policy?")
run_agent("How long does shipping take?")
run_agent("My order is completely wrong and I am very frustrated, please help!")
run_agent("What is the capital of France?")