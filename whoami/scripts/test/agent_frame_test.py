from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.chat_models import ChatOpenAI

# llm = ChatOpenAI(
#             # model="qwen2:latest", 
#             model="qwen2.5:7b-instruct",
#             temperature=0.0, 
#             api_key="ollama", 
#             base_url="http://localhost:11434/v1",
#         )

# @tool
# def multiply(a: int, b: int):
#     """Multiply two numbers."""
#     return a * b

if __name__ == '__main__':
    
    @tool
    def multiply(a: int, b: int):
        """Multiply two numbers."""
        return a * b
    
    print(multiply.name)
    print(multiply.description)
    print(multiply.args)
