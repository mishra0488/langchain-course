# In THis section we are writing tertieval grader chain, which is goin to use structured output from 
# LLM and turning it into pydantic object that will have the information whether the doc is relevant or not
# and if the doc is not relavant, we want to filter it out and keep only the docs which are relevant to the question
# and if not all docs are relevant then we want to mark web search flag to be true.
# SO THIS CHAIN GOINT TO RECIEVE INPUT THE ORIGINAL QUESTION AND THE RETRIEVE DOC AND DETeRMINE WHETHER DOCIS RELEVANT OR NOT
# WE WILL RUN THIS CHAIN FOR EACH DOC WE RETRIEVE. 


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# using "with_structured_output" method
# Langchain will use func calling and for every LLM call we make, we will return pydantic object
# and LLm is going to return in the schema we want.
structured_llm_grader = llm.with_structured_output(GradeDocuments)


# System prompt we will send to LLm
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# CREATE THE CHAIN - 
# It is going to take grade prompt and pipe it into LLM with structured output
retrieval_grader = grade_prompt | structured_llm_grader