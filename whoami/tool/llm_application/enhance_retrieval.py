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
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter


from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig

from whoami.tool.base.base_tool import BaseTool



class EnhanceRetrieval(BaseTool):
    """
    已完成简单的检索增强
    后续加入混合检索逻辑
    还有现有的检索存在漏洞，如果问题没有对应的检索向量，总是会返回固定数量的内容，这不是我想要的，因为会影响最终的回答质量

    Args:
        BaseTool (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    data_dir: Optional[str] = None
    index_dir:  Optional[str] = None
    embed_model: HuggingFaceEmbedding = None
    llm: Optional[Union[OllamaLLM, Ollama]] = None
    node_parser: SimpleNodeParser = None
    static_index: Optional[VectorStoreIndex] = None  # 用于存储静态索引
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
        
        
        # 层次化分块，效果太差
        # self.node_parser = HierarchicalNodeParser.from_defaults(
        #     chunk_sizes=[1024, 512, 128],
        #     chunk_overlap=20,
        # )
        
        # 如果索引目录中已有索引，则加载它
        # if os.path.exists(self.index_dir):
        #     self.index = self.load_index()
        # else:
        #     self.index = None
        # 加载或创建静态索引
        # 初始化静态索引
        self.static_index = None
        try:
            self.initialize_static_index()
        except Exception as e:
            self.logger.error(f"初始化静态索引失败: {str(e)}")
    
    
    
    
    
    def initialize_static_index(self):
        """初始化静态索引 - 加载现有索引或创建新索引"""
        # 确保索引目录存在
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir, exist_ok=True)
            self.logger.info(f"创建索引目录: {self.index_dir}")

        # 尝试加载现有的索引
        try:
            docstore_path = os.path.join(self.index_dir, "docstore.json")
            if os.path.exists(docstore_path):
                self.logger.info(f"加载现有索引: {self.index_dir}")
                # 使用LlamaIndex的API加载索引
                storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
                self.static_index = load_index_from_storage(storage_context)
                return
        except Exception as e:
            self.logger.warning(f"加载现有索引失败: {str(e)}")
            # 如果索引损坏，清理目录
            try:
                import shutil
                shutil.rmtree(self.index_dir)
                os.makedirs(self.index_dir, exist_ok=True)
            except Exception as clean_err:
                self.logger.error(f"清理索引目录失败: {str(clean_err)}")

        # 创建新索引
        self.logger.info("创建新索引...")
        
        # 使用LlamaIndex的SimpleDirectoryReader加载文档

        
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            self.logger.warning(f"数据目录不存在：创建初始化目录 - {self.data_dir}")
        else:
            try:
                # 使用LlamaIndex的文档加载器，它会自动处理各种文件格式
                # 默认支持：.txt, .pdf, .docx, .pptx, .csv, 等等
                reader = SimpleDirectoryReader(
                    input_dir=self.data_dir,
                    recursive=True,  # 递归处理子目录
                    filename_as_id=True,  # 使用文件名作为ID
                    required_exts=[".txt", ".md", ".csv", ".json", ".html", ".pdf", ".docx"]  # 只处理这些扩展名的文件
                )
                static_documents = reader.load_data()
                self.logger.info(f"成功加载 {len(static_documents)} 个文档")
                
                # 如果未找到文档，创建一个空文档
                if len(static_documents) == 0:
                    static_documents = [Document(text="初始化文档", metadata={"source": "init", "type": "static"})]
                    self.logger.warning("未找到任何文档，使用初始化文档创建索引")
            except Exception as load_err:
                self.logger.error(f"加载文档失败: {str(load_err)}")
                # 创建一个空文档
                static_documents = [Document(text="初始化文档", metadata={"source": "init", "type": "static"})]
        
        # 为文档添加类型元数据
        for doc in static_documents:
            doc.metadata["type"] = "static"
        
        try:
            # 解析文档为节点
            nodes = self.node_parser.get_nodes_from_documents(static_documents)
            self.logger.info(f"已创建 {len(nodes)} 个节点")
            
            # 创建索引
            self.static_index = VectorStoreIndex(nodes)
            
            # 持久化索引
            self.logger.info(f"正在将索引持久化到 {self.index_dir}...")
            self.static_index.storage_context.persist(persist_dir=self.index_dir)
            self.logger.info("索引创建和持久化成功")
        except Exception as create_err:
            self.logger.error(f"创建静态索引失败: {str(create_err)}")
            self.static_index = None
    
    
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
    
    
    def retrieve(
        self, 
        text_list: List[Dict[str, str]], 
        top_k: int = 3, 
        query: str = None,
        static_flag: int = 1
    ):
        if query is None:
            raise ValueError("query must not be null!")
        
        results = []
        # 从动态text_list创建索引并检索
        if text_list:
            try:
                dynamic_index = self.store_index(text_list)
                if dynamic_index:
                    dynamic_retriever = dynamic_index.as_retriever(similarity_top_k=top_k)
                    dynamic_nodes = dynamic_retriever.retrieve(query)
                    results.extend(dynamic_nodes)
                    self.logger.info(f"dynamic_nodes: ------------------------ {dynamic_nodes}")
            except Exception as e:
                self.logger.error(f"从动态文本检索失败: {str(e)}")
        
        
        # 从静态索引中检索
        if static_flag != 0:
            try:
                if self.static_index is None:
                    # 如果静态索引未初始化，尝试重新初始化
                    self.logger.info("静态索引未初始化，尝试重新初始化...")
                    self.initialize_static_index()
                    
                if self.static_index:
                    static_retriever = self.static_index.as_retriever(similarity_top_k=top_k)
                    static_nodes = static_retriever.retrieve(query)
                    results.extend(static_nodes)
                    self.logger.info(f"static_nodes: ------------------------ {static_nodes}")
            except Exception as static_err:
                self.logger.error(f"从静态索引检索失败: {str(static_err)}")
            
        
        # 如果两个源都没有结果
        if not results:
            self.logger.warning("No retrieval results found!")
            return []
        
        results.sort(key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else getattr(x, 'similarity', 0.0), reverse=True)
        
        # 只返回top_k个结果
        return results[:top_k] if len(results) > top_k else results
        
        
        """
        if not text_list:
            self.logger.warning("text_list is null!")
            return []
        
        index = self.store_index(text_list)
        
        if index is None:
            raise ValueError("索引尚未创建或加载，请先调用 store_index 方法")
        
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        return nodes
        """
    
    
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
            stream_flag: int = 1,
            database_retrieval_data: List[Dict[str, str]] = None
        ):
        default_prompt = """
            你是舜熙科技的客服助手。请基于以下上下文信息、数据库检索结果和用户历史会话信息，以专业、简洁的口吻回答用户问题。
            
            上下文信息:
            {context}
            
            指导原则:
            1. 直接回答问题，不要包含分析过程
            2. 如果上下文中没有相关信息，请礼貌表示不知道
            3. 只回答与舜熙科技相关的问题
            4. 保持礼貌友好的专业客服语气
            
            当前系统时间：
            {current_time}
            
            {enhance_role}：
            {database_enhance_prompt}
            
            
            历史会话消息：
            {message_history}

            用户当前问题：
            {rewritten_query}
            """ if prompt is None else prompt
            
        rewritten_query = query if rewritten_query is None else rewritten_query
        if retrieval_flag:
            # 只要进行检索，都需要调用store vector
            nodes = self.retrieve(text_list=text_list, top_k=top_k, query=query)
            context_texts = [node.node.text for node in nodes]
            context = "无可用上下文信息" if not context_texts else "\n\n".join(context_texts)
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
        else:
            try:
                text_list = str(text_list) if text_list is not None else "无可用上下文信息"
            except Exception as e:
                raise ValueError(f"fail to init text_list! {str(e)}") from e
        # text_list为检索/不检索的上下文信息
        enhance_role = "数据库检索内容/检索结果" if (enhance_role is None or enhance_role == "") else enhance_role
        database_enhance_prompt = str(database_retrieval_data) if database_retrieval_data is not None else "无可用信息"
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        message = ""
        if prompt is None:
            message = default_prompt.format(
                context=text_list,
                current_time=current_time,
                enhance_role=enhance_role,
                database_enhance_prompt=database_enhance_prompt,
                message_history=message_history,
                rewritten_query=rewritten_query
            )
        else:
            message = prompt + f"\n\n上下文信息：\n{text_list}" + f"\n\n当前系统时间：\n{current_time}" + f"\n\n{enhance_role}：\n{database_enhance_prompt}" + f"\n\n历史会话消息：\n{message_history}" + f"\n\n用户当前问题：\n{rewritten_query}"
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
    text_list = [
        {"123": "我是卫宇涛，我28，我来自山西运城"}, 
        {"456": "我是卫小涛，30岁，来自山西运城"}, 
        {"789": "我是卫jin涛，30岁，来自山西运城"},
        {"1011": "我是卫jin涛，30岁，来自山西运城"},
        {"1012": "我是卫jin涛，30岁，来自山西运城"},
        {"1013": "我是卫jin涛，30岁，来自山西运城"},
        {"1014": "我是卫jin涛，30岁，来自山西运城"},
    ]
    text_list = None
    #  使用asyncio运行异步函数
    async def main():
        async_generator = enhance_._run(text_list=text_list, query='你是谁？', top_k=3, retrieval_flag=1)
        full_result = ""
        async for chunk in async_generator:
            print("Received chunk:", chunk)  # Optional: print each chunk as received
            full_result += chunk
    
        print("Complete result:", full_result)
    
    # 运行异步主函数
    asyncio.run(main())


