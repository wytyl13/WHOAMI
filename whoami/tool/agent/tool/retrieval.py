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
import asyncio
import jieba
import Stemmer
import traceback
from enum import Enum
from pydantic import BaseModel, Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from rank_bm25 import BM25Okapi


from whoami.tool.agent.base_tool import tool


class RetrievalSchema(BaseModel):
    retrieval_word: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="检索关键词，一般为用户的问题"
    )



class StrEnum(str, Enum):
    def __str__(self) -> str:
        # overwrite the __str__ method to implement enum_instance.attribution == enum_instance.attribution.value
        return self.value
    
    def __repr__(self) -> str:
        return f"'{str(self)}'"



class RankType(StrEnum):
    """Rank type"""
    reciprocal_rank_fusion = "reciprocal_rank_fusion"


  

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
    static_bm25_retriever: Optional[BM25Retriever] = None
    
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
        if 'static_bm25_retriever' in kwargs:
            self.static_bm25_retriever = kwargs.pop('static_bm25_retriever')
        
            
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
            self._initialize_bm25_index()
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
    
    
    def _initialize_bm25_index(self):
        """初始化BM25索引"""
        try:
            # 如果向量索引存在，从中获取节点
            if self.static_index and hasattr(self.static_index, "docstore"):
                self.static_nodes = list(self.static_index.docstore.docs.values())
                self.logger.info(f"从向量索引获取了 {len(self.static_nodes)} 个节点用于BM25索引")
                
                # 对节点文本进行分词
                self.static_corpus = []
                for node in self.static_nodes:
                    if hasattr(node, 'text') and node.text:
                        tokens = list(jieba.cut(node.text))
                        self.static_corpus.append(tokens)
                    else:
                        # 如果节点没有文本，使用空列表
                        self.static_corpus.append([])
                
                # 创建BM25索引
                self.static_bm25_index = BM25Okapi(self.static_corpus)
                self.logger.info("BM25索引创建成功")
            else:
                self.logger.warning("静态向量索引不存在，无法创建BM25索引")
                self.static_bm25_index = None
        except Exception as e:
            self.logger.error(f"创建BM25索引失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.static_bm25_index = None
    
    
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
    
    
    def store_bm25_index(self, text_list: List[Dict[str, str]]):
        # 处理动态文本
        pass
    
    
    def vector_retrieval(
        self, 
        retrieval_word: str = None, 
        text_list: List[Dict[str, str]] = None, 
        top_k: int = 3,
        static_flag: int = 1
    ):
        """
        向量检索
        
        参数:
            retrieval_word: 检索关键词
            text_list: 动态文本列表，格式为[{key1: text1}, {key2: text2}, ...]
            top_k: 返回的最大结果数
            static_flag: 是否检索静态索引(1表示检索，0表示不检索)
        
        返回:
            检索结果列表
        """
        results = []
        if text_list:
            try:
                dynamic_index = self.store_index(text_list)
                if dynamic_index:
                    dynamic_retriever = dynamic_index.as_retriever(similarity_top_k=top_k)
                    dynamic_nodes = dynamic_retriever.retrieve(retrieval_word)
                    results.extend(dynamic_nodes)
                    # self.logger.info(f"dynamic_nodes: ------------------------ {dynamic_nodes}")
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
                    # self.logger.info(f"static_nodes: ------------------------ {static_nodes}")
            except Exception as static_err:
                self.logger.error(f"从静态索引检索失败: {str(static_err)}")
        return results
    
    
    def keyword_retrieval_llamaindex(self, 
        retrieval_word: str = None, 
        text_list: List[Dict[str, str]] = None, 
        top_k: int = 3,
        static_flag: int = 1
    ):
        """
        使用BM25算法进行关键词检索
        
        参数:
            retrieval_word: 检索关键词
            text_list: 动态文本列表，格式为[{key1: text1}, {key2: text2}, ...]
            top_k: 返回的最大结果数
            static_flag: 是否检索静态索引(1表示检索，0表示不检索)
        
        返回:
            检索结果列表
        """
        query_tokens = list(jieba.cut(retrieval_word))
        chinese_query = " ".join(query_tokens)
        results = []
        if text_list:
            try:
                # 将文本列表转换为Document对象
                documents = []
                for item in text_list:
                    source_key = list(item.keys())[0]
                    text_content = list(item.values())[0]
                    
                    # 对中文文本进行分词处理
                    text_tokens = list(jieba.cut(text_content))
                    processed_text = " ".join(text_tokens)  # 用空格连接分词结果
                    
                    documents.append(Document(
                        text=processed_text,  # 使用处理后的文本
                        metadata={"source": source_key, "original_source": source_key, "original_text": text_content}
                    ))
                
                if documents:
                    try:
                        # 创建临时文档存储
                        from llama_index.core.storage.docstore import SimpleDocumentStore
                        from llama_index.core.node_parser import SentenceSplitter
                        
                        # 解析为节点
                        splitter = SentenceSplitter(chunk_size=512)
                        nodes = self.node_parser.get_nodes_from_documents(documents)
                        
                        # 创建docstore
                        docstore = SimpleDocumentStore()
                        docstore.add_documents(nodes)
                        
                        # 创建BM25检索器
                        try:
                            stemmer = Stemmer.Stemmer("porter")
                            dynamic_bm25_retriever = BM25Retriever.from_defaults(
                                docstore=docstore,
                                similarity_top_k=top_k,
                                stemmer=stemmer,
                                language="english",  # 我们已经用jieba处理了中文
                            )
                        except Exception as stemmer_err:
                            self.logger.warning(f"无法使用stemmer创建BM25检索器: {str(stemmer_err)}")
                            dynamic_bm25_retriever = BM25Retriever.from_defaults(
                                docstore=docstore,
                                similarity_top_k=top_k
                            )
                        
                        # 创建查询包
                        query_bundle = QueryBundle(query_str=chinese_query)
                        # 执行检索
                        dynamic_nodes = dynamic_bm25_retriever.retrieve(query_bundle)
                        self.logger.info(f"动态BM25检索结果数: {len(dynamic_nodes)}")
                        
                        # 记录分数
                        for node in dynamic_nodes:
                            self.logger.info(f"分数: {node.score}, 文本: {node.node.text[:50]}...")
                        
                        results.extend(dynamic_nodes)
                    except Exception as retriever_err:
                        self.logger.error(f"创建动态BM25检索器失败: {str(retriever_err)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
            except Exception as e:
                self.logger.error(f"从动态文本进行BM25检索失败: {str(e)}")

                self.logger.error(traceback.format_exc())
    
        # 处理静态索引
        if static_flag != 0 and hasattr(self, 'static_bm25_retriever') and self.static_bm25_retriever is not None:
            try:
                # 创建查询包
                query_bundle = QueryBundle(query_str=chinese_query)
                
                # 执行检索
                static_nodes = self.static_bm25_retriever.retrieve(query_bundle)
                self.logger.info(f"静态BM25检索结果数: {len(static_nodes)}")
                
                # 记录分数
                for node in static_nodes:
                    self.logger.info(f"分数: {node.score}, 文本: {node.node.text[:50]}...")
                
                results.extend(static_nodes)
            except Exception as e:
                self.logger.error(f"从静态索引进行BM25检索失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 确保所有结果都有分数
        for result in results:
            if not hasattr(result, 'score') or result.score is None:
                result.score = 0.0
        
        # # 按分数排序
        # results.sort(key=lambda x: x.score, reverse=True)
        
        # # 返回前top_k个结果
        # return results[:top_k] if len(results) > top_k else results
        return results
    
    
    def keyword_retrieval(self, 
        retrieval_word: str = None, 
        text_list: List[Dict[str, str]] = None, 
        top_k: int = 3,
        static_flag: int = 1
    ):
        """
        使用BM25算法进行关键词检索
        """
        
        
        results = []
        
        # 处理动态文本
        if text_list:
            try:
                # 准备文档和对应的原始文本
                docs = []
                original_docs = []
                for item in text_list:
                    source_key = list(item.keys())[0]
                    text_content = list(item.values())[0]
                    docs.append(text_content)
                    original_docs.append({
                        "text": text_content,
                        "source": source_key
                    })
                
                # 使用jieba分词
                tokenized_corpus = [list(jieba.cut(doc)) for doc in docs]
                tokenized_query = list(jieba.cut(retrieval_word))
                
                # 创建BM25模型并计算分数
                bm25 = BM25Okapi(tokenized_corpus)
                scores = bm25.get_scores(tokenized_query)
                
                # 创建结果
                for i, score in enumerate(scores):
                    doc = original_docs[i]
                    results.append(NodeWithScore(
                        node=Document(
                            text=doc["text"],
                            metadata={"source": doc["source"], "original_source": doc["source"]}
                        ),
                        score=score
                    ))
                
                self.logger.info(f"动态BM25检索结果数: {len(results)}")
                # for i, result in enumerate(results):
                #     self.logger.info(f"{i+1}. 分数: {result.score}, 文本: {result.node.text}")
                    
            except Exception as e:
                self.logger.error(f"从动态文本进行BM25检索失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 静态索引检索
        if static_flag != 0 and self.static_index is not None:
            try:
                tokenized_query = list(jieba.cut(retrieval_word))
                
                # 使用预构建的BM25索引计算分数
                scores = self.static_bm25_index.get_scores(tokenized_query)
                
                # 创建结果
                for i, score in enumerate(scores):
                    if i < len(self.static_nodes):  # 防止索引越界
                        results.append(NodeWithScore(
                            node=self.static_nodes[i],
                            score=score
                        ))
                
                self.logger.info(f"静态BM25检索结果数: {len(self.static_nodes)}")
            except Exception as e:
                self.logger.error(f"从静态索引进行BM25检索失败: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # 排序
        # results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前top_k个结果
        # return results[:top_k]
        return results
    
    
    def reciprocal_rank_fusion(self, vector_results, bm25_results, k=60, top_k=3):
        """
        实现RRF (Reciprocal Rank Fusion)
        
        参数:
            - vector_results: 向量检索结果
            - bm25_results: BM25检索结果
            - k: 常数，控制排名得分衰减 (通常为60)
            - top_k: 返回的结果数量
        """
        # 结果映射
        results_map = {}
        
        # 处理向量检索结果
        for i, node in enumerate(vector_results):
            key = node.node.id_ if hasattr(node.node, 'id_') else node.node.text
            rank = i + 1  # 排名从1开始
            results_map[key] = {
                'node': node.node,
                'score': 1.0 / (k + rank)
            }
        
        # 处理BM25检索结果
        for i, node in enumerate(bm25_results):
            key = node.node.id_ if hasattr(node.node, 'id_') else node.node.text
            rank = i + 1  # 排名从1开始
            
            if key in results_map:
                # 累加RRF分数
                results_map[key]['score'] += 1.0 / (k + rank)
            else:
                results_map[key] = {
                    'node': node.node,
                    'score': 1.0 / (k + rank)
                }
        
        # 生成最终排序结果
        final_results = [
            NodeWithScore(node=item['node'], score=item['score'])
            for item in results_map.values()
        ]
        
        # 按分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:top_k]
    
    
    def rerank(
        self, 
        query: str = None, 
        vector_results: List = None,
        bm25_results: List = None,
        top_k: int = 3,
        rank_type: Optional[str] = "reciprocal_rank_fusion"
    ):
        """重排序

        Args:
            query (str, optional): _description_. Defaults to None.
            candidates (List, optional): _description_. Defaults to None.
            top_k (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        if rank_type == RankType.reciprocal_rank_fusion:
            try:
                return self.reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
            except Exception as e:
                raise ValueError(f"Fail to exec reciprocal_rank_fusion function! {str(e)}") from e
            
        else:
            # 普通重排序
            # 合并混合检索候选结果并去重
            candidate_map = {}
            for result in vector_results + bm25_results:
                key = result.node.id_ if hasattr(result.node, 'id_') else result.node.text
                if key not in candidate_map:
                    candidate_map[key] = result
        
            candidates = list(candidate_map.values())
            try:
                candidates.sort(key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else getattr(x, 'similarity', 0.0), reverse=True)
                # 只返回top_k个结果
                return candidates[:top_k] if len(candidates) > top_k else candidates
            except Exception as e:
                raise ValueError(f"Fail to exec rerank function! {str(e)}") from e
    
    
    async def execute(
        self, 
        text_list: List[Dict[str, str]], 
        top_k: int = 3, 
        retrieval_word: str = None,
        static_flag: int = 1,
        rank_type: Optional[str] = "reciprocal_rank_fusion"
    ):
        if retrieval_word is None:
            raise ValueError("retrieval_word must not be null!")
        
        
        vector_results = self.vector_retrieval(retrieval_word=retrieval_word, text_list=text_list, top_k=top_k, static_flag=static_flag)
        bm25_results = self.keyword_retrieval(retrieval_word=retrieval_word, text_list=text_list, top_k=top_k, static_flag=static_flag)
        
        
        return self.rerank(
            query=retrieval_word, 
            vector_results=vector_results, 
            bm25_results=bm25_results,
            top_k=top_k,
            rank_type=rank_type
        )
        
        



if __name__ == '__main__':
    text_list = [
        {"123": "我是卫宇涛，我28，我来自山西运城"}, 
        {"456": "我们公司地址在山西省运城市万荣县科创城"}, 
        {"789": "我是卫jin涛，30岁，来自山西运城"},
        {"1011": "我是卫jin涛，30岁，来自山西运城"},
        {"1012": "我是卫jin涛，30岁，来自山西运城"},
        {"1013": "我是卫jin涛，30岁，来自山西运城"},
        {"1014": "我是卫jin涛，30岁，来自山西运城"},
    ]
    retrieval = Retrieval()
    async def main():
        nodes = await retrieval.execute(text_list=text_list, retrieval_word='母乳的成分变化', top_k=5)
        print(nodes)
    asyncio.run(main())
    