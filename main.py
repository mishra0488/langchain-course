import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
# ClientSession provides framwork for python application to act as an mcp client
from mcp.client.stdio import stdio_client
# stdio_client - connect with mcp server through transport i/o

load_dotenv()

llm = ChatOpenAI()

# This variable going to hold all information on how to run our mcp server
stdio_server_params = StdioServerParameters(
    command="python",
    args=["C:/Automation/langchain-course/servers/match_server.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (read,write):
        # create client session - as every client connect through clientsession - will do the communication b/w client & server  
        async with ClientSession(read_stream=read, write_stream=write) as session:
            # initialize the session
            await session.initialize()
            print("session initialized")
            tools = await load_mcp_tools(session)

            agent = create_agent(llm,tools)  # this will langgraph react agent
            # ainvoke bec we are async mode
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 54 + 2 * 3?")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
