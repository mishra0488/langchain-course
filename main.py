from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()



# if using openAi

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


# with free tool ollama ---> since do not support search tool or tavitly search therefore no tools used
# llm = ChatOllama(model="gemma3:270m")
# agent = create_agent(model=llm)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in tokyo")})
    print(result)
    

if __name__ == "__main__":
    main()
