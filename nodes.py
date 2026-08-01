from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# This is the node (function) which is going to recieve state, which is message state which is adictionary
# that has key of messages
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}


# This is the second node
tool_node = ToolNode(tools)