from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


# recieve state and return dictionary what to update in state
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    # extract question from current state
    question = state["question"]

    # do all the symantic search and get us all relevant doc
    documents = retriever.invoke(question)
    # and as a return we want to update field doc in our current state with retrieve docs
    return {"documents": documents, "question": question}