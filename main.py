import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from dtos import AgentResponse

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, response_format=AgentResponse)
agent = create_agent(model=llm, tools=[search])


def main():
    print("Hello from linkedin-job-agent!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Procure por 3 vagas de trabalho remoto no Brasil para AI Engineer no Linkedin que busquem experiência em LangChain. Liste somente vagas ativas."
            )
        }
    )
    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
