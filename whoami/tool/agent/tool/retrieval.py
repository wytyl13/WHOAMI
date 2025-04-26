#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/12 23:17:28
@Author : weiyutao
@File : retrieval.py
"""

from typing import (
    Optional,
    Type,
    Any,
    List,
    Dict
)
import os
from pydantic import BaseModel, Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
import asyncio


from whoami.tool.agent.base_tool import tool


class RetrievalSchema(BaseModel):
    retrieval_word: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="检索关键词，一般为用户的问题"
    )


@tool
class Retrieval:
    """
    Retrieval tool for any function what need to retrieval.
    """    
    end_flag: int = 0
    args_schema: Type[BaseModel] = RetrievalSchema
    data_dir: Optional[str] = None
    index_dir:  Optional[str] = None
    embed_model: Optional[HuggingFaceEmbedding] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    node_parser: Optional[Any] = None
    static_index: Optional[VectorStoreIndex] = None
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'data_dir' in kwargs:
            self.data_dir = kwargs.pop('data_dir')
        if 'index_dir' in kwargs:
            self.index_dir = kwargs.pop('index_dir')
        if 'embed_model' in kwargs:
            self.embed_model = kwargs.pop('embed_model')
        if 'chunk_size' in kwargs:
            self.chunk_size = kwargs.pop('chunk_size')
        if 'chunk_overlap' in kwargs:
            self.chunk_overlap = kwargs.pop('chunk_overlap')
        if 'node_parser' in kwargs:
            self.node_parser = kwargs.pop('node_parser')
        if 'static_index' in kwargs:
            self.static_index = kwargs.pop('static_index')
        
            
        # 初始化一些东西
        self.chunk_size = 512 if self.chunk_size is None else self.chunk_size
        self.chunk_overlap = 20 if self.chunk_overlap is None else self.chunk_overlap  
        self.data_dir = "/work/ai/WHOAMI/retrieval_data" if self.data_dir is None else self.data_dir
        self.index_dir = "/work/ai/WHOAMI/retrieval_storage" if self.index_dir is None else self.index_dir
        self.embed_model = HuggingFaceEmbedding(model_name='/work/ai/WHOAMI/whoami/models/embedding/bge-large-zh-v1.5')
    

        # 使用setting创建服务上下文
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
    

        # 创建自定义节点解析器
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,       # 自定义块大小
            chunk_overlap=50      # 自定义重叠大小
        ) if self.node_parser is None else self.node_parser
        
        
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
    
            
    async def execute(
        self, 
        text_list: List[Dict[str, str]], 
        top_k: int = 3, 
        retrieval_word: str = None,
        static_flag: int = 1
    ):
        if retrieval_word is None:
            raise ValueError("retrieval_word must not be null!")
        
        results = []
        # 从动态text_list创建索引并检索
        if text_list:
            try:
                dynamic_index = self.store_index(text_list)
                if dynamic_index:
                    dynamic_retriever = dynamic_index.as_retriever(similarity_top_k=top_k)
                    dynamic_nodes = dynamic_retriever.retrieve(retrieval_word)
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
                    static_nodes = static_retriever.retrieve(retrieval_word)
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
        



if __name__ == '__main__':
    text_list = [
        {"123": "我是卫宇涛，我28，我来自山西运城"}, 
        {"456": "我是卫小涛，30岁，来自山西运城"}, 
        {"789": "我是卫jin涛，30岁，来自山西运城"},
        {"1011": "我是卫jin涛，30岁，来自山西运城"},
        {"1012": "我是卫jin涛，30岁，来自山西运城"},
        {"1013": "我是卫jin涛，30岁，来自山西运城"},
        {"1014": "我是卫jin涛，30岁，来自山西运城"},
    ]
    retrieval = Retrieval()
    print(retrieval)
    async def main():
        nodes = await retrieval.execute(text_list=text_list, retrieval_word='你是谁？')
        print(nodes)
    asyncio.run(main())
    