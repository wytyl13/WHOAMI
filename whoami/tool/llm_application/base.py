import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from whoami.configs.llm_config import LLMConfig
from pathlib import Path
from llama_index.core.query_engine import SQLTableRetrieverQueryEngine

llm=Ollama(model="qwen2.5:7b-instruct", request_timeout=360.0)

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = AgentWorkflow.from_tools_or_functions(
    [multiply],
    llm=llm,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))

async def main():
    # Run the agent
    response = await llm.astream_complete("who am i")
    # # print(str(response))
    async for chunk in response:
        print(chunk.delta, end="", flush=True)
db_connection = "mysql+pymysql://(root):{2xryuf@I73T}@{192.168.0.10}:{3366}/{shunxikeji}"
sql_query_engine = SQLTableRetrieverQueryEngine(db_connection, table_retriever=)

user_query = "用户的问题"
retrieved_data = sql_query_engine.query(user_query)
# Run the agent
if __name__ == "__main__":
    # response = llm.whoami_text("What is 1234 * 4567?", stream=True)
    # for chunk in response:
    #     print(chunk.delta, end="", flush=True)
    print(retrieved_data)