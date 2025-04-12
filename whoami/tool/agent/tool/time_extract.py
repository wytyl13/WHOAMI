from typing import (
    Any
)
from datetime import datetime
from pathlib import Path

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import InfoExtract


@tool
class TimeExtract(InfoExtract):
    """
    因为继承的父类已经实现了抽象方法，并且符合该工具的要求，因此不再实现抽象方法
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')
            
        if 'default_extract_result' in kwargs:
            self.default_extract_result = kwargs.get('default_extract_result')
            
if __name__ == '__main__':
    from whoami.llm_api.ollama_llm import OllamaLLM
    from whoami.configs.llm_config import LLMConfig
    import asyncio
    llm = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    time_extract = TimeExtract(llm=llm)
    print(time_extract)
    async def main(question: str):
        result = await time_extract.execute(question=question)
        print(result)
        return result
    asyncio.run(main("汇报下最近1个月的睡眠情况"))