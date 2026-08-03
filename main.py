from typing import TypedDict, Annotated
# typedicts is a type dictionary which creates a structured dictionary with hints for the keys, 
# and will be used here to define state schema
# This is needed bec lang graph would require typed state definitions to know which data flows through and out of graph
# Annotated help us to add metadata to those type hints

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage is abstract base class for all message types in langchain 
# HumanMessage to represent a message from user

from langgraph.graph import END, StateGraph
# END marks the termination of graph with end node
# StateGraph is the main class for building stateful graph, goimg to be data structure holds the info of execution
# and stores intermediate results llm resonses and everything else.
# It act as input to every node >> and when node performs task, it updates the state

from langgraph.graph.message import add_messages
# add_messages - Entire goal of this func is to ensure new msgea are appended to the existing conversations histry
# instead of replacing it.

from chains import generate_chain, reflect_chain

# messages here is a list of base msg objects
# and annotation here is metadata that will tell lang graph how to handle state updates
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# name of the nodes
REFLECT = "reflect"
GENERATE = "generate"

# 1st node - and input to this is staTE OF Type MessageGraph (when nothing is generated, first execution will be user input)
# later it will also have critique as message
# retrning dictionary, with key as messages and value as we get from running chain which is an ai message
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


# returning critique and intentinally labeling it as human message, bec we want LLM to think that critique is from human
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


# object of the state graph and schema is MessageGraph
# to tell lang graph what would be state how to update it
builder = StateGraph(state_schema=MessageGraph)

# Create nodes
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# To decide this as first node, so by default it starts from node start, after that first node will be generate
builder.set_entry_point(GENERATE)

# Func for conditional edge
# From generate node to >> reflect node >> genearate >> either to reflect or to END node
# input of this func is state and output is string, name of node
# THis func will be called everytime after we run generation node
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# Create conditional edge , 1st input is generate and 2nd is where to go (reflect or end)
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
# 2nd edge from reflection node to gerate node
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)