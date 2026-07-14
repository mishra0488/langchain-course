from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


load_dotenv()

# class represents source of the answer
class Source(BaseModel):
    """Schema for source used by agents"""
    url:str = Field(description="The URL of the source")

# list of sources and an answer
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources to generate the answer")

# if using openAi
llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


# with free tool ollama ---> since do not support search tool or tavitly search therefore no tools used
# llm = ChatOllama(model="gemma3:270m")
# agent = create_agent(model=llm)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages":HumanMessage(
                content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print(result)
    

if __name__ == "__main__":
    main()
