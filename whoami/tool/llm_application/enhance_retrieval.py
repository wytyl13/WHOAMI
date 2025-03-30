import os
# from llama_index import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     ServiceContext,
#     StorageContext,
#     load_index_from_storage
# )

from typing import (
    Optional,
    Union,
    List,
    Dict
)
from pathlib import Path
import asyncio
import copy
from datetime import datetime, timedelta


from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig

from whoami.tool.base.base_tool import BaseTool



class EnhanceRetrieval(BaseTool):
    
    data_dir: Optional[str] = None
    index_dir:  Optional[str] = None
    embed_model: HuggingFaceEmbedding = None
    llm: Optional[Union[OllamaLLM, Ollama]] = None
    node_parser: SimpleNodeParser = None
    
    def __init__(
        self,
        data_dir="/work/ai/WHOAMI/retrieval_data",
        index_dir="/work/ai/WHOAMI/retrieval_storage",
        embed_model="text-embedding-ada-002",
        llm: Optional[Union[OllamaLLM, Ollama]] = None,
        chunk_size=1024,
        chunk_overlap=20
    ):
        super().__init__()
        self.data_dir = data_dir
        self.index_dir = index_dir
        
        self.embed_model = HuggingFaceEmbedding(model_name='/work/ai/WHOAMI/whoami/models/embedding/bge-large-zh-v1.5')
        self.llm = llm
        
        # 2. 创建服务上下文
        # 使用Settings替代ServiceContext
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # 3. 创建自定义节点解析器（可以调整参数）
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,       # 自定义块大小
            chunk_overlap=50      # 自定义重叠大小
        )
        
        # 如果索引目录中已有索引，则加载它
        # if os.path.exists(self.index_dir):
        #     self.index = self.load_index()
        # else:
        #     self.index = None
    
    def store_index(self, text_list: List[Dict[str, str]]):
        try:
            documents = []
            for item in text_list:
                source_key = list(item.keys())[0]
                text_content = list(item.values())[0]
                documents.append(Document(text=text_content, metadata={"source": source_key, "original_source": source_key}))
                
            # 解析文档为节点
            nodes = self.node_parser.get_nodes_from_documents(documents)

            # 从节点创建索引
            index = VectorStoreIndex(nodes)
        except Exception as e:
            self.logger.error(f"Fail to exec store index function, {str(e)}")
            return None
        return index
    
    def retrieve(self, text_list: List[Dict[str, str]], top_k: int = 3, query: str = None):
        if query is None:
            raise ValueError("query must not be null!")
        
        if not text_list:
            self.logger.warning("text_list is null!")
            return []
        
        index = self.store_index(text_list)
        
        if index is None:
            raise ValueError("索引尚未创建或加载，请先调用 store_index 方法")
        
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        return nodes
    
    async def _run(
            self, 
            text_list: List[Dict[str, str]], 
            top_k: int = 3, 
            query: str = None, 
            message_history: List[Dict[str, str]] = None,
            retrieval_flag: Optional[bool] = False,
            enhance_role: Optional[str] = None,
            rewritten_query: Optional[str] = None,
            prompt: Optional[str] = None,
            stream_flag: int = 1
        ):
        prompt = f"""
            你是舜熙科技的客服助手。请基于以下用户历史会话信息，以专业、简洁的口吻回答用户问题。

            指导原则:
            1. 直接回答问题，不要包含分析过程
            2. 如果上下文中没有相关信息，请礼貌表示不知道
            3. 只回答与舜熙科技相关的问题
            4. 保持礼貌友好的专业客服语气
            5. 必要时可以引导用户访问舜熙科技官网: https://shunxikj.com/
            """ if prompt is None else prompt
        rewritten_query = query if rewritten_query is None else rewritten_query
        if retrieval_flag:
            nodes = self.retrieve(text_list=text_list, top_k=top_k, query=query)
            context_texts = [node.node.text for node in nodes]
            context = "\n\n".join(context_texts)
            sources = []
            for node in nodes:
                # 优先使用我们存储的original_source，然后是source
                source = node.node.metadata.get('original_source')
                if not source:
                    source = node.node.metadata.get('source')
                if not source:
                    # 如果都没有，使用节点ID作为后备
                    source = str(node.node.id_)
                sources.append(source)
            text_list = context
            prompt = f"""
            你是舜熙科技的客服助手。请基于以下上下文信息，以专业、简洁的口吻回答用户问题。

            上下文信息:
            {context}

            指导原则:
            1. 直接回答问题，不要包含分析过程
            2. 如果上下文中没有相关信息，请礼貌表示不知道
            3. 只回答与舜熙科技相关的问题
            4. 保持礼貌友好的专业客服语气
            5. 必要时可以引导用户访问舜熙科技官网: https://shunxikj.com/

            用户问题: {rewritten_query}
            """
        else:
            try:
                text_list = str(text_list)
            except Exception as e:
                raise ValueError(f"fail to init text_list! {str(e)}") from e
        enhance_role = "数据库检索内容/检索结果" if (enhance_role is None or enhance_role == "") else enhance_role
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        message = prompt + f"当前系统时间：{current_time}" + f"\n\n{enhance_role}: {text_list}" + f"\n\n历史会话消息：{message_history}" + f"\n\n用户当前问题：{rewritten_query}"
        namespace_message_history = [{"role": "user", "content": message}]
        # namespace_message_history = copy.deepcopy(message_history)
        # if text_list != '[]':
        #     namespace_message_history.append({"role": enhance_role, "content": text_list})
        # namespace_message_history.append({"role": "user", "content": rewritten_query})
        self.logger.info(f"namespace_message_history --------------------------------------------------  {namespace_message_history}")
        
        if stream_flag == 1:
            # 使用流式输出接口
            chat_stream = self.llm._whoami_text_stream(messages=namespace_message_history, timeout=30, user_stop_words=[])
            if not chat_stream:
                self.logger.error("Stream is empty or None!")
                yield "Error: Could not obtain streaming response from LLM."
                return
            try:
                async for chunk in chat_stream:
                    # 只处理非空内容
                    if chunk:
                        # 返回当前块
                        yield chunk
            except Exception as e:
                self.logger.error(f"处理流时出错: {str(e)}")
                yield f"Error: {str(e)}"
        else:
            # 使用非流式输出接口
            try:
                response = await self.llm._whoami_text(messages=namespace_message_history, timeout=30, user_stop_words=[])
                yield response
                return  # 一次性返回完整响应后结束
            except Exception as e:
                self.logger.error(f"非流式处理时出错: {str(e)}")
                yield f"Error: {str(e)}"
                return

if __name__ == "__main__":
    llm = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
    enhance_ = EnhanceRetrieval(llm=llm)
    text_list = [{"123": "我是卫宇涛，我28，我来自山西运城"}, {"456": "我是卫小涛，30岁，来自山西运城"}]
    #  使用asyncio运行异步函数
    async def main():
        result = await enhance_._run(text_list=text_list, query='卫宇涛年龄')
        print("最终结果:", result)
    
    # 运行异步主函数
    asyncio.run(main())


