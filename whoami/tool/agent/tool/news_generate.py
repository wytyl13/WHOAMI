#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/04 17:57
@Author  : weiyutao
@File    : news_generate.py
未解决bug
1、google检索的图片为svg，无法正常插入word，应该先转换为png。这块没做
2、小标题对应的生成内容偏题：比如：裕丰昌携手海南椰岛，构建“健康酱酒+康养体验”业态主题对应的 --- （裕丰昌-椰岛合作）这个小标题生成的内容偏题，可能的情况是这个主题基本没有检索到内容。
3、第三个小标题生成不精确：
    QuestMobile：46岁+男性网络活跃用户达2.26亿   第三个小标题为：46岁男性网络活跃用户   其实应该是QuestMobile
    北京大家小家等合作提升银发医疗保障体系  第三个小标题不应该简单为北京
    
"""

from typing import (
    Optional,
    Dict,
    Any,
    overload
)
import asyncio
from pathlib import Path
import json
import requests
import os
from urllib.parse import urlparse
from datetime import datetime
import ast
import time
import re

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import GoogleSearch
from whoami.tool.search.google_search_provider import GoogleSearchProvider
from whoami.tool.agent.tool.age_club_content import AgeClubContent
from whoami.tool.agent.tool.relevance_extract import RelevanceExtract
from whoami.tool.agent.tool.retrieval import Retrieval
from whoami.tool.agent.tool.subject_extract import SubjectExtract
from whoami.tool.agent.tool.search_term_expansion import SearchTermExpansion
from whoami.utils.utils import Utils
from whoami.tool.agent.tool.title_content_generate import TitleContentGenerate
from whoami.tool.agent.tool.info_extract import InfoExtract
from whoami.tool.agent.tool.key_entity_extract import KeyEntityExtract
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool.key_abridge import KeyAbridge


utils = Utils()
save_path = "/work/ai/WHOAMI/whoami/out/news_generate"

@tool
class NewsGenerate(InfoExtract):
    google_search_provider: Optional[GoogleSearchProvider] = None
    relevance_extract: Optional[RelevanceExtract] = None
    subject_extract: Optional[SubjectExtract] = None
    retrieval: Optional[Retrieval] = None
    search_term_expansion: Optional[SearchTermExpansion] = None
    title_content_generate: Optional[TitleContentGenerate] = None
    key_entity_extract: Optional[KeyEntityExtract] = None
    enhance_retrieval: Optional[EnhanceRetrieval] = None
    key_abridge: Optional[KeyAbridge] = None
    
    
    @overload
    def __init__(
        self, 
        google_search: Optional[GoogleSearch] = None,
        relevance_extract: Optional[RelevanceExtract] = None,
        subject_extract: Optional[SubjectExtract] = None,
        retrieval: Optional[Retrieval] = None,
        search_term_expansion: Optional[SearchTermExpansion] = None,
        title_content_generate: Optional[TitleContentGenerate] = None,
        key_entity_extract: Optional[KeyEntityExtract] = None,
        enhance_retrieval: Optional[EnhanceRetrieval] = None,
        key_abridge: Optional[KeyAbridge] = None
    ):
        ...


    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if 'google_search_provider' in kwargs:
            self.google_search_provider = kwargs.pop('google_search_provider')
        if 'relevance_extract' in kwargs:
            self.relevance_extract = kwargs.pop('relevance_extract')
        if 'subject_extract' in kwargs:
            self.subject_extract = kwargs.pop('subject_extract')
        if 'retrieval' in kwargs:
            self.retrieval = kwargs.pop('retrieval')
        if 'search_term_expansion' in kwargs:
            self.search_term_expansion = kwargs.pop('search_term_expansion')
        if 'title_content_generate' in kwargs:
            self.title_content_generate = kwargs.pop('title_content_generate')
        if 'key_entity_extract' in kwargs:
            self.key_entity_extract = kwargs.pop('key_entity_extract')
        if 'enhance_retrieval' in kwargs:
            self.enhance_retrieval = kwargs.pop('enhance_retrieval')
        if 'key_abridge' in kwargs:
            self.key_abridge = kwargs.pop('key_abridge')


    def sanitize_filename(self, filename, max_length=100):
        """
        清理文件名，移除不合法字符
        
        Args:
            filename (str): 原始文件名
            max_length (int): 文件名最大长度，默认100
        
        Returns:
            str: 清理后的安全文件名
        """
        if not filename:
            return "image"
        
        # 分离文件名和扩展名
        name_part, ext_part = os.path.splitext(filename)
        
        # 清理文件名部分
        # 移除或替换不合法字符 (Windows + Linux + macOS 不兼容字符)
        illegal_chars = r'[<>:"|?*\\/!@#$%^&()+=\[\]{};\'`~]'
        clean_name = re.sub(illegal_chars, '_', name_part)
        
        # 移除连续的下划线、空格和点
        clean_name = re.sub(r'[_\s.]+', '_', clean_name)
        
        # 移除开头和结尾的下划线
        clean_name = clean_name.strip('_')
        
        # 确保不为空
        if not clean_name:
            clean_name = "image"
        
        # 限制文件名长度（保留扩展名的空间）
        max_name_length = max_length - len(ext_part)
        if len(clean_name) > max_name_length:
            clean_name = clean_name[:max_name_length]
        
        # 清理扩展名（移除特殊字符，只保留点和字母数字）
        if ext_part:
            clean_ext = re.sub(r'[^.a-zA-Z0-9]', '', ext_part)
            # 确保扩展名不会太长
            if len(clean_ext) > 6:  # 如 .jpeg 最长5个字符
                clean_ext = '.jpg'  # 默认扩展名
        else:
            clean_ext = '.jpg'  # 默认扩展名
        
        return clean_name + clean_ext


    def search_images(self, query, filename=None, save_path="/work/ai/WHOAMI/whoami/out/news_generate"):         
        url = "https://www.googleapis.com/customsearch/v1"              
        params = {             
            'q': query,
            'cx': "4095b4ae3b7704074",
            'key': "AIzaSyBMaQpsm0RBsUhc9zCokInfCjKPpvS9wiY",
            'searchType': 'image',
            'num': 5,   # 改为5个结果，增加成功率
            'safe': 'active',
            'imgSize': 'medium',
        }              
        
        try:             
            response = requests.get(url, params=params)             
            data = response.json()             
            
            if 'items' not in data or len(data['items']) == 0:                 
                return {'success': False, 'error': '没有找到相关图片'}              
            
            # 尝试多个结果，直到找到一个可用的
            for image_item in data['items']:
                image_url = image_item['link']             
                image_title = image_item.get('title', query)
                
                # 简单验证URL格式
                if not image_url.startswith(('http://', 'https://')):
                    continue  # 跳过无效URL
                
                try:
                    # 添加浏览器头部，避免被屏蔽
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    img_response = requests.get(image_url, timeout=30, headers=headers)             
                    img_response.raise_for_status()                          
                    
                    # 生成文件名
                    if filename is None:                 
                        parsed_url = urlparse(image_url)                 
                        ext = os.path.splitext(parsed_url.path)[1]                 
                        if not ext:                     
                            ext = '.jpg'
                        safe_title = "".join(c for c in image_title if c.isalnum() or c in (' ', '-', '_')).strip()                 
                        if not safe_title:                     
                            safe_title = query                                  
                        filename = f"{safe_title[:50]}{ext}"      
                        filename = self.sanitize_filename(filename)      
                    else:
                        filename = self.sanitize_filename(filename)
                    os.makedirs(save_path, exist_ok=True)                    
                    full_path = os.path.join(save_path, filename)                      
                    
                    with open(full_path, 'wb') as f:                 
                        f.write(img_response.content)                          
                    self.logger.info(f"full_path, {full_path}")
                    return {                 
                        'success': True,                 
                        'file_path': full_path,                 
                        'filename': filename,                 
                        'image_url': image_url,                 
                        'title': image_title,                 
                        'size': len(img_response.content)             
                    }
                    
                except:
                    continue  # 这个图片失败了，试下一个
            
            # 所有图片都失败了
            return {'success': False, 'error': '所有图片下载都失败了'}
            
        except requests.exceptions.RequestException as e:             
            return {'success': False, 'error': f'网络请求错误: {str(e)}'}         
        except Exception as e:             
            return {'success': False, 'error': f'未知错误: {str(e)}'}


    async def execute(
        self, 
        query: Optional[str] = None,
        query_abstract: Optional[str] = None,
        query_expansion: int = 0
    ):
        original_query = query
        query = utils.clean_text(query)
        


        if query_expansion:
            """
            检索词扩充
            """
            search_term_result = await self.search_term_expansion.execute(question=query)
            print(search_term_result)
            node_texts = []
            for i in range(2):
                param = {
                    "query": search_term_result["queries"][i]
                }
                status, result = self.google_search_provider(**param)
                if not status:
                    raise ValueError(f"fail to exec google search! {result}")
                text_list = [{item["link"]: item["fetch_url_content"]} for item in result]
                retrieval_result = await self.retrieval.execute(text_list=text_list, retrieval_word=query, static_flag=0, top_k=3)
                node_text = self.retrieval.safe_extract_text(retrieval_result)
                node_texts.extend(node_text)
        else:
            param = {
                "query": query
            }
            status, result = self.google_search_provider(**param)
            if not status:
                raise ValueError(f"fail to exec google search! {result}")
            text_list = [{item["link"]: item["fetch_url_content"]} for item in result]
            retrieval_result = await self.retrieval.execute(text_list=text_list, retrieval_word=query, static_flag=0, top_k=3)
            node_texts = self.retrieval.safe_extract_text(retrieval_result)
        subject_extract_content = {"需要回答的问题": "请参考以下主题、主题摘要和检索内容，还有参考上面的规则要求，完成你的角色任务", "主题": query, "主题摘要": query_abstract, "检索内容": node_texts}
        result = await self.subject_extract.execute(question=str(subject_extract_content))
        generate_title = list(result.values())
        generate_title = ["详情", "影响"] if not generate_title else generate_title
        if len(generate_title) == 1:
            generate_title.append("事件影响")
        generate_content = await self.title_content_generate.execute(
            title1=generate_title[0],
            title2=generate_title[1],
            subject=query,
            abstract=query_abstract,
            search_content=node_texts
        )
        result = self.parse_json_response(generate_content)
        if isinstance(result, str):
            result = json.loads(result) 
        print(result)
        
        key_entity = await self.key_entity_extract.execute(input=query)
        param = {
            "query": key_entity
        }
        status, result_ = self.google_search_provider(**param)
        if not status:
            raise ValueError(f"fail to exec google search! {result_}")
        text_list = [{item["link"]: item["fetch_url_content"]} for item in result_]
        print(text_list)
        if not text_list:
            key_entity = await self.key_abridge.execute(question=key_entity, str_flag=1)
            param = {
                "query": key_entity
            }
            status, result_ = self.google_search_provider(**param)
            if not status:
                raise ValueError(f"fail to exec google search! {result_}")
            text_list = [{item["link"]: item["fetch_url_content"]} for item in result_]
        
        result__ = None
        
        # enhance_prompt = """
        # 您是专业咨询顾问。基于以下信息源，为用户提供准确、专业的解答。

        # ## 信息源
        # 上下文信息: {context}
        # 已有信息：{information}
        # 数据库检索结果: {database_enhance_prompt}  
        # 历史会话记录: {message_history}
        # 当前系统时间: {current_time}

        # ## 输出要求
        # 1. **内容要求**: 直接回答用户问题，重点突出核心信息
        # 2. **格式要求**: 输出单段连贯内容，避免分节分段
        # 3. **长度控制**: 控制在150-200字以内
        # 4. **专业标准**: 使用准确的行业术语，确保信息客观性
        # 5. **内容聚焦**: 仅回答用户关心的具体问题，避免无关建议

        # ## 禁止输出
        # - 分析过程和方法论说明
        # - 信息来源和数据局限性声明  
        # - 额外的用户体验建议
        # - 与问题无关的延伸内容

        # 用户问题: {question}

        # 请直接提供简洁专业的回答。
        # """
        
        enhance_prompt = """
        您是专业咨询顾问。基于以下信息源，为用户提供准确、专业的解答。

        ## 信息源
        上下文信息: {context}
        已有信息：{information}
        数据库检索结果: {database_enhance_prompt}  
        历史会话记录: {message_history}
        当前系统时间: {current_time}

        ## 重复性检查（核心要求）
        1. **避免重复**: 仔细检查已有信息，严禁输出已提及的内容
        2. **增量信息**: 优先提供已有信息中未涵盖的新信息或不同角度
        3. **补充价值**: 如已有信息充分，提供深度补充或细节完善
        4. **无新信息处理**: 若确无新信息可补充，简要说明"相关信息已为您提供"

        ## 输出要求
        1. **内容要求**: 直接回答用户问题，重点突出核心信息
        2. **格式要求**: 输出单段连贯内容，避免分节分段
        3. **长度控制**: 控制在150-200字以内
        4. **专业标准**: 使用准确的行业术语，确保信息客观性
        5. **内容聚焦**: 仅回答用户关心的具体问题，避免无关建议

        ## 禁止输出
        - 已有信息中包含的任何具体内容
        - 分析过程和方法论说明
        - 信息来源和数据局限性声明  
        - 额外的用户体验建议
        - 与问题无关的延伸内容

        用户问题: {question}

        请直接提供简洁专业的回答，确保与已有信息不重复，不要有和回答主题不相关的提醒、解释和客套话描述内容。
        """
        
        enhance_prompt = enhance_prompt.replace("information", str(result))
        async for result_item in self.enhance_retrieval.execute(text_list=text_list, top_k=3, question=key_entity + "简介及事件、历程", static_flag=0, prompt=enhance_prompt):
            result__ = result_item
            break  # Get the first (and only) result
        result[key_entity] = result__
        # if generate_title:
        #     for title in generate_title:
        #         generate_content = await self.title_content_generate.execute(
        #             title=title,
        #             subject=query,
        #             abstract=query_abstract,
        #             search_content=node_texts
        #         )
        #         result_dict[title] = generate_content
        
        result_dict = {}
        search_image_result = self.search_images(query=key_entity)
        image_path = search_image_result["file_path"] if "file_path" in search_image_result else "none.png"
        result_dict[original_query] = image_path
        result_dict["content"] = result
        return result_dict
    
    
class JsonToMd:
    
    def __init__(self):
        pass
    
    def generate_markdown_from_json(self, json_data, date_str=None):
        """
        将JSON数据转换为Markdown格式
        
        Args:
            json_data: 输入的JSON数据（可以是字符串或已解析的Python对象）
            date_str: 可选的日期字符串，如果不提供则使用当天日期
        
        Returns:
            str: 生成的Markdown文本
        """
        
        # 如果输入是字符串，先解析JSON
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # 获取日期，如果没有提供则使用当天日期
        if date_str is None:
            date_str = datetime.now().strftime("%Y年%m月%d日")
        
        # 开始构建Markdown内容
        markdown_lines = []
        
        # 添加一级标题
        markdown_lines.append(f"# {date_str}养老行业每天热点回顾\n")
        
        # 遍历每个新闻项目
        for i, item in enumerate(data):
            if i > 0:
                markdown_lines.append("---")  # 分隔线
                markdown_lines.append("")  # 空行
            # 获取第一个键作为二级标题，对应的值作为图片路径
            first_key = list(item.keys())[0]
            image_path = item[first_key]
            
            # 添加二级标题
            markdown_lines.append(f"## {first_key}")
            
            # 添加图片（如果有图片路径）
            if image_path:
                markdown_lines.append(f"![图片](<{image_path}>)")
            
            markdown_lines.append("")  # 空行
            
            # 获取content字典
            content_dict = item.get('content', {})
            
            # 遍历content中的每个项目作为三级标题
            for subtitle, content in content_dict.items():
                markdown_lines.append(f"### {subtitle}")
                markdown_lines.append(f"{content}")
                markdown_lines.append("")  # 空行
            
            # markdown_lines.append("")  # 在每个二级标题之间添加额外空行
        
        return "\n".join(markdown_lines)

    def save_markdown_to_file(self, json_data, filename=None, date_str=None):
        """
        将JSON数据转换为Markdown并保存到文件
        
        Args:
            json_data: 输入的JSON数据
            filename: 保存的文件名，如果不提供则自动生成
            date_str: 可选的日期字符串
        
        Returns:
            str: 保存的文件名
        """
        
        # 生成Markdown内容
        markdown_content = self.generate_markdown_from_json(json_data, date_str)
        
        # 如果没有提供文件名，自动生成
        if filename is None:
            if date_str is None:
                date_str = datetime.now().strftime("%Y年%m月%d日")
            filename = f"{date_str}养老行业热点回顾.md"
        
        # 保存到文件
        with open(f"/work/ai/WHOAMI/whoami/out/news_generate/{filename}", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return filename    
    



if __name__ == '__main__':
    
    from whoami.llm_api.ollama_llm import OllamaLLM
    from whoami.configs.llm_config import LLMConfig
    llm = OllamaLLM(
        config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')),
        temperature=0.0
    )
    retrieval = Retrieval(chunk_size=200, chunk_overlap=20)
    relevance_extract = RelevanceExtract(llm = llm)
    subject_extract = SubjectExtract(llm = llm)
    search_term_expansion = SearchTermExpansion(llm = llm)
    google_search_provider = GoogleSearchProvider(search_config_path="/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml")
    age_club_content = AgeClubContent()
    title_content_generate = TitleContentGenerate(llm = llm)
    key_entity_extract = KeyEntityExtract(llm=llm)
    enhance_retrieval = EnhanceRetrieval(llm=llm)
    key_abridge = KeyAbridge(llm=llm)
    news_generate = NewsGenerate(
        google_search_provider=google_search_provider, 
        relevance_extract=relevance_extract, 
        subject_extract=subject_extract,
        retrieval=retrieval,
        search_term_expansion=search_term_expansion,
        title_content_generate=title_content_generate,
        key_entity_extract=key_entity_extract,
        enhance_retrieval=enhance_retrieval, 
        key_abridge=key_abridge
    )
    async def main():
        results = await age_club_content.execute()
        result_list = []
        for item in results:
            result = await news_generate.execute(query = item["title"], query_abstract = item["content"])
            result_list.append(result)
        return result_list
    
    result = asyncio.run(main())
    json_to_md = JsonToMd()
    json_to_md.save_markdown_to_file(result)

    print(result)
    
    # json_content = """[{'小橙集团康力元参编民政部“毫米波雷达监测报警器”新标准': '/work/ai/WHOAMI/whoami/out/news_generate/首发小橙集团完成新一轮数千万元战略融资加速数字化医护养老布局.jpg', 'content': {'参编贡献': '小橙集团康力元作为民政部《监测和定位辅助器具 毫米波雷达监测报警器》行业新标的参编单位之一，在此次标准制定过程中，充分发挥其技术优势与专业力量。通过细化具体内容，为标准的科学性和可行性提供了重要支持。', '标准影响': '该新标准的发布实施将对毫米波雷达监测报警器的技术规范和市场应用产生深远影响。它不仅提升了产品的安全性和可靠性，还推动了相关行业的标准化进程，有助于提高整体服务质量与用户体验。', '小橙集团': '小橙集团专注于大健康领域，尤其在综合为老服务方面有深厚积累。旗下品牌涵盖养老护理、康复辅具租赁、适老化改造等全方位居家养老服务，并提供智慧养老产品和医养机构嵌入式代运营等专业解决方案。近期，小橙集团康力元作为民政部《监测和定位辅助器具 毫米波雷达监测报警器》行业新标的参编单位之一，为标准的科学性和可行性提供了重要支持。此外，小橙集团还参与了中国职工发展基金会职工家庭健康保险保障公益项目，并签约提供专业赔付服务。'}}, {'小橙集团参与“职工家庭防癌抗癌保障卡”项目': '/work/ai/WHOAMI/whoami/out/news_generate/首发小橙集团完成新一轮数千万元战略融资加速数字化医护养老布局.jpg', 'content': {'项目签约': '近日，小橙集团受邀参加在全国总工会机关举行的中国职工发展基金会“职工家庭健康保险保障公益项目捐赠仪式暨2025年职工家庭防癌抗癌保障服务启动”活动。作为此次公益项目的捐赠和服务支持方之一，在现场小橙集团与中国职工发展基金会签约，深度参与中国职工发展基金会“职工家庭防癌抗癌保障卡”项目，并为护理服务责任提供专业赔付。', '保障服务': '小橙集团将以此次项目为契机，携手全体联合承保机构，为广大职工家庭提供更具温度、更为专业的保障服务。通过这一项目，小橙集团将致力于提升职工家庭的健康保险水平，确保在面对癌症等重大疾病时能够获得及时有效的医疗和护理支持。', '小橙集团': '小橙集团近期受邀参加中国职工发展基金会的捐赠仪式，作为公益项目的捐赠和服务支持方之一，与中国职工发展基金会签约，并参与“职工家庭防癌抗癌保障卡”项目。此举旨在提升职工家庭健康保险水平，确保在面对重大疾病时获得及时有效的医疗和护理支持。'}}, {'共比邻旗下“邻家优选”合作西安百跃羊乳': '/work/ai/WHOAMI/whoami/out/news_generate/音画制作软件免费下载安装-共比邻音画剪辑软件下载v510 安卓版-3673.png', 'content': {'合作背景': '近日，上海共比邻旗下社区健康生活品牌“邻家优选”与西安百跃羊乳集团达成合作。双方将携手打造“乡村振兴+银发健康”模式，为老年群体提供健康营养解决方案。“邻家优选”深耕社区场景，致力于为广大中老年群体提供优质的健康食品与便民服务。', '项目规划': '合作将共同开发适合老年人群的定制化羊乳产品，满足银发群体对营养均衡、易吸收的健康需求。双方计划通过深入研究老年人的身体特点和营养需求，推出一系列符合市场需求的产品和服务，以提升老年居民的生活质量。', '共比邻': '上海共比邻旗下社区健康生活品牌“邻家优选”近期与西安百跃羊乳集团达成合作，共同探索适合老年人群的定制化羊乳产品。此次合作旨在通过深入研究老年群体的身体特点和营养需求，提供健康解决方案，助力提升他们的生活质量。“邻家优选”深耕社区场景，致力于为广大中老年群体提供优质的健康食品与便民服务。'}}, {'苏宁易购旗下适老化品牌已覆盖31个城市': '/work/ai/WHOAMI/whoami/out/news_generate/Suningcom - Wikipedia.png', 'content': {'适老化品牌覆盖': '苏宁易购旗下品牌在多个城市推出了针对老年人的服务项目，包括但不限于智能家居、健康监测和紧急呼叫系统。这些服务旨在提升老年人的生活质量，并提供便捷的居家养老服务。', '创新服务模式': '为了更好地服务于社区居民，苏宁易购在全国范围内新增了500家帮客便民服务站，覆盖超过5万个市民生活广场和居民社区。通过线下门店与线上平台相结合的方式，为消费者提供家电安装、维修及清洗等一站式服务。', '苏宁易购': '苏宁易购是一家综合性零售企业，旗下品牌在多个城市推出了针对老年人的服务项目，涵盖智能家居、健康监测和紧急呼叫系统。此外，苏宁易购在全国范围内新增了500家帮客便民服务站，为消费者提供家电安装、维修及清洗等一站式服务。2024年5月，由于苏宁集团违约未能偿还对橡树资本的欠债，后者接管了苏宁持有的国际米兰股份，成为俱乐部的实际持有人。同年，苏宁易购与美团宣布深化合作，上线空调最快2小时即送即装服务。'}}, {'河北省长期照护师报考工作展开': '/work/ai/WHOAMI/whoami/out/news_generate/新闻库_青年之声.png', 'content': {'考试公告发布': '近日，河北省发布了长期照护师职业技能等级认定考试公告。该考试将在2025年于承德护理职业学院举行。此次考试旨在进一步提升河北省长期照护服务的专业化水平。', '报考条件解析': '报考长期照护师（五级/初级工）没有学历、工龄等方面的限制，这使得更多有志之士能够参与到这一领域中来。值得注意的是，这是继江苏之后全国第二批开展的长期照护师职业技能等级认定考试工作。', '河北省长期照护师报考工作': '河北省近期发布了关于长期照护师职业技能等级认定考试的公告，该考试计划于2025年在承德护理职业学院举行。此次考试旨在提升长期照护服务的专业化水平。值得注意的是，报考长期照护师（五级/初级工）没有学历、工龄等方面的限制，这使得更多有志之士能够参与到这一领域中来。这是继江苏之后全国第二批开展的此类考试工作。'}}, {'河南省首笔养老产业贷款落地': '/work/ai/WHOAMI/whoami/out/news_generate/河南省- 维基百科自由的百科全书.png', 'content': {'首笔养老贷款': '6月3日，记者在河南省人民政府和河南日报获悉，在中国人民银行河南省分行指导下，邮储银行新乡市分行近日主动对接中原农谷康养社区项目，为该项目授信5.5亿元，并发放3600万元养老产业贷款。这笔贷款是服务消费与养老再贷款政策出台后全省落地的首笔养老产业贷款。', '项目实施分析': '该贷款资金专项用于支持中原农谷康养社区设施建设，具体包含4栋服务型公寓、1栋护理型公寓、1栋康养服务中心及相关配套设施。此外，自中国人民银行宣布设立5000亿元服务消费与养老再贷款政策出台以来，中国人民银行河南省分行迅速行动，召开全省金融机构落实一揽子货币政策措施工作会议，加强政策传导和解读，引导全省金融机构紧抓机遇靠前发力，加快推动包含服务消费与养老再贷款在内的一揽子货币政策落地见效。', '河南省': '河南省位于中国中部，省会郑州。全省面积167062平方公里，辖17个地级行政区，包括郑州市、开封市等。截至2024年末，常住人口约9853万人。河南历史悠久，是华夏文明的重要发源地之一，拥有众多历史文化遗产，如殷墟、龙门石窟等世界文化遗产。近年来，河南省积极推进经济发展与产业升级，特别是在养老服务领域取得新进展，如首笔养老产业贷款的发放，支持了中原农谷康养社区项目的发展。'}}, {'总投资2.8亿，上海奉贤新城养老院开工': '/work/ai/WHOAMI/whoami/out/news_generate/奉贤新城喜获优秀今年上半年度市容环境质量社会公众满意度测评结果.gif', 'content': {'开工典礼': '6月4日上午，总投资2.8亿元的奉贤新城养老院改扩建工程正式开工。区委书记袁泉宣布开工，并与区委副书记区长王益群等共同为项目培土奠基。区人大常委会副主任唐瑛副区长田哲区政协副主席姚朝晖出席，副区长李慧致辞。', '项目规划': '该项目作为奉贤区2025年重大民生工程之一，选址于原奉贤区福利院旧址（东至古华路，南临江海河，西接南横泾，北抵环城南路），用地面积1.23万平方米，总建筑面积约3.3万平方米。项目建成后将成为集综合养护、医养融合、智慧养老、实训基地于一体的复合型养老服务机构，共设616张床位，包含123张认知症专护床位，预计于2027年底全面完工。', '上海奉贤新城': '奉贤新城养老院改扩建工程已于2025年6月4日开工，总投资2.8亿元。项目选址于原福利院旧址，预计2027年底完工，将提供616张床位，包括123张认知症专护床位，集综合养护、医养融合、智慧养老及实训基地于一体。此外，近期还举行了东方美谷生命信使基因药物创新产业基地开园仪式和第五届海聚英才全球创新创业大赛等重要活动。'}}, {'东软集团等成立康养产业公司': '/work/ai/WHOAMI/whoami/out/news_generate/东软集团山东信息科技有限公司 领英.jpg', 'content': {'公司成立概况': '近日，沈阳盛情康养产业有限公司正式成立。法定代表人为王星辉，注册资本为1000万人民币。该公司经营范围广泛，包括养老服务、人工智能应用软件开发、智能家庭消费设备销售等。股权数据显示，东软集团等共同持股该公司。', '康养产业布局': '沈阳盛情康养产业有限公司的成立标志着东软集团在康养产业领域迈出重要一步。其业务涵盖了养老服务和人工智能技术的应用，旨在通过智能化手段提升服务质量和效率。未来，公司将继续探索更多创新应用，推动康养产业的发展。', '东软集团': '相关信息显示，东软集团（Neusoft）是一家中国领先的软件与信息技术服务企业。公司成立于1991年，并于2000年在大连创立。东软主要业务涵盖医疗健康、智慧城市、智能汽车互联等领域。近年来，东软持续推动技术创新和市场拓展，在国内外均取得显著成绩。具体事件及历程方面，东软经历了多次重要发展节点，包括上市、并购整合等关键阶段，但详细历史信息未能从现有数据中获取。'}}]"""
    # data = ast.literal_eval(json_content)
