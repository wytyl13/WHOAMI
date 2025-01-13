
from pathlib import Path


from whoami.core.runnable import Runable
from whoami.llm_api.ollama_llm import OllamLLM
from whoami.configs.llm_config import LLMConfig

llm = OllamLLM(LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")))

class RunnableImpl(Runable[str, str]):
    
    def invoke(self, input, config = None, **kwargs):
        return llm.whoami(input)

    
    
    