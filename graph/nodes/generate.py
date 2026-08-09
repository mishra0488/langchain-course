from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


# This node will take the question and take docs from our state and run the chain
def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}