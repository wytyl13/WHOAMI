import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi import APIRouter
from dataclasses import dataclass, field
import uvicorn
from llama_index.llms.ollama import Ollama
from typing import (
    Optional,
    Dict
)
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware

from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.utils.R import R
from whoami.tool.llm_application.query_classifier import QueryClassifier
from whoami.tool.llm_application.chat_sys_test import ChatSys
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.llm_application.sx_coversation_history import SxConversationHistory
from whoami.utils.utils import Utils

llm=Ollama(model="qwen2.7-7b-Instruction-sx-8epochs-258:latest", request_timeout=360.0)
llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
sql_config_path = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'
# 创建ChatSys实例，传入数据库提供者
sql_provider = SqlProvider(model=SxConversationHistory, sql_config_path=sql_config_path)
chat_sys = ChatSys(llm=llm, sql_provider=sql_provider)

app = FastAPI()
# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端源
    # 如果需要允许所有源，可以使用 ["*"]
    # allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有头
)
# 添加路径前缀
prefix_router = APIRouter(prefix="/ai/chat_sys")
@dataclass
class RequestDataChat:
    question: str = None
    conversation_id: str = None
    user_id: str = None  
    messages: Optional[list[Dict[str, str]]] = None


@dataclass
class RequestDataConversationHistory:
    user_id: str = None
    conversation_id: Optional[str] = None


@dataclass
class TruncateConversationHistory:
    user_id: str = None
    conversation_id: Optional[str] = None


utils = Utils()
STOPWORDS_1 = set()
with open("/work/ai/WHOAMI/whoami/tool/llm_application/stop_words.txt", "r", encoding="utf-8") as f:
    for line in f:
        STOPWORDS_1.add(line.strip())

def group_by_user_id(data):
    """
    Group messages by user_id in the provided data.
    
    Args:
        data (list): List of dictionaries containing user_id and messages.
        
    Returns:
        dict: Dictionary with user_id as keys and their messages combined.
    """
    result = {}
    for item in data:
        conversation_id = item.get("conversation_id")
        role = item.get("role", '')
        content = item.get("content", '')
        # messages = [{"conversation": conversation_id, "messages": [{"role": item["role"], "content": item["content"]} for item in messages]}]
        if conversation_id not in result:
            # Create new entry for this user_id
            result[conversation_id] = {
                "conversation_id": conversation_id,
                "messages": []
            }
        result[conversation_id]["messages"].append({"role": role, "content": content})
        
    # Convert the dictionary to a list if needed
    return list(result.values())


def test_rag():
    @prefix_router.post('/chat_health_report')
    async def chat_health_report(request_data: RequestDataChat):
        # logger.info(request_data)
        try:
            question = request_data.question
            conversation_id = request_data.conversation_id
            user_id = request_data.user_id
            messages = request_data.messages
        except Exception as e:
            return R.fail(f"传参错误！{request_data}")

        if question is None or question == "":
            return R.fail("question must not be null")
            
        # question = utils.remove_stopwords(question, stop_words=STOPWORDS_1)
        # question = '舜熙科技' + question if '舜熙' not in question else question
        
        if conversation_id is None or conversation_id == "":
            return R.fail("conversation_id must not be null")
        
        if user_id is None or user_id == "":
            return R.fail("user_id must not be null")
        
        messages = [] if messages is None or messages == "" or not messages else messages
        if not messages:
            result = sql_provider.get_record_by_condition(condition={"user_id": user_id, "conversation_id": conversation_id})
            if result:
                result = group_by_user_id(result)
                messages = result[0]["messages"][-10:] # 仅使用最后10条数据
            
        # 创建异步生成器以便与StreamingResponse一起使用
        async def response_generator():
            async for text_chunk in chat_sys._run(messages_history=messages, question=question, direct_flag=0, stream_flag=1, user_id='13D4F349200080712111959C07'):
                yield f"data: {text_chunk}\n\n"
                await asyncio.sleep(0.01)  # 小延迟，避免过快消耗

        


        # 创建一个包装生成器，在流完成后保存数据
        async def wrapped_generator():
            try:
                # 手动消费内部生成器并传递每个块
                async for chunk in response_generator():
                    yield chunk
            except Exception as e:
                print(f"流处理出错: {str(e)}")
                raise
            finally:
                # 无论成功还是失败，确保在流完成后保存数据
                print("流式响应完成，准备保存对话到数据库...")
                try:
                    await chat_sys.save_qa_to_db(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=question
                    )
                    print(f"对话保存完成，full_response长度: {len(''.join(chat_sys.full_response))}")
                except Exception as e:
                    print(f"保存对话失败: {str(e)}")

        # 返回流式响应
        response =  StreamingResponse(
            wrapped_generator(),
            media_type="text/event-stream"
        )
        
        # 简单添加CORS头
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response

    @prefix_router.post('/get_conversation_history')
    async def get_conversation_history(request_data: RequestDataConversationHistory):
        result = []
        try:
            user_id = request_data.user_id
            conversation_id = request_data.conversation_id
        except Exception as e:
            return R.fail(f"传参错误！{request_data}")
        
        if user_id is None or user_id == "":
            return R.fail(f"user_id must not be null!")
        
        if conversation_id is None or conversation_id == "":
            result = sql_provider.get_record_by_condition({"user_id": user_id})
            result = group_by_user_id(result)
        else:
            result = sql_provider.get_record_by_condition({"user_id": user_id, "conversation_id": conversation_id})
            if not result:
                result = []
            else:
                result = [{"conversation": conversation_id, "messages": [{"role": item["role"], "content": item["content"]} for item in result]}]
        return R.success(result)


    @prefix_router.post('/truncate_conversation_history')
    def truncate_conversation_history(request_data: TruncateConversationHistory):
        try:
            user_id = request_data.user_id
            conversation_id = request_data.conversation_id
        except Exception as e:
            return R.fail(f"传参错误！{request_data}")
        if user_id is None or user_id == "":
            return R.fail(f"user_id must not be null!")
        
        condition = {"user_id": user_id} if (conversation_id is None or conversation_id == "") else {"user_id": user_id, "conversation_id": conversation_id}
        
        try:
            result = sql_provider.delete_records_by_condition(condition=condition)
        except Exception as e:
            return R.fail(f"Fail to truncate conversation history! {str(e)}")
        return R.success(f"Successfully truncated conversation history, {result}")
    
    
    @prefix_router.post('/chat_health_report_gw')
    async def chat_health_report_gw(request_data: RequestDataChat):
        # logger.info(request_data)
        try:
            question = request_data.question
            conversation_id = request_data.conversation_id
            user_id = request_data.user_id
            messages = request_data.messages
        except Exception as e:
            return R.fail(f"传参错误！{request_data}")

        if question is None or question == "":
            return R.fail("question must not be null")
            
        # question = utils.remove_stopwords(question, stop_words=STOPWORDS_1)
        # question = '舜熙科技' + question if '舜熙' not in question else question
        if conversation_id is None or conversation_id == "":
            conversation_id = user_id
        
        if conversation_id is None or conversation_id == "":
            return R.fail("conversation_id must not be null")
        
        if user_id is None or user_id == "":
            return R.fail("user_id must not be null")
        
        messages = [] if messages is None or messages == "" or not messages else messages
        if not messages:
            result = sql_provider.get_record_by_condition(condition={"user_id": user_id, "conversation_id": conversation_id})
            if result:
                result = group_by_user_id(result)
                messages = result[0]["messages"][-20:] # 仅使用最后10条数据
            
        # 创建异步生成器以便与StreamingResponse一起使用
        async def response_generator():
            async for text_chunk in chat_sys._run(messages_history=messages, question=question, direct_flag=1):
                yield f"data: {text_chunk}\n\n"
                await asyncio.sleep(0.01)  # 小延迟，避免过快消耗

        # 创建一个包装生成器，在流完成后保存数据
        async def wrapped_generator():
            try:
                # 手动消费内部生成器并传递每个块
                async for chunk in response_generator():
                    yield chunk
            except Exception as e:
                print(f"流处理出错: {str(e)}")
                raise
            finally:
                # 无论成功还是失败，确保在流完成后保存数据
                print("流式响应完成，准备保存对话到数据库...")
                try:
                    await chat_sys.save_qa_to_db(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=question
                    )
                    print(f"对话保存完成，full_response长度: {len(''.join(chat_sys.full_response))}")
                except Exception as e:
                    print(f"保存对话失败: {str(e)}")

        # 返回流式响应
        response =  StreamingResponse(
            wrapped_generator(),
            media_type="text/event-stream"
        )
        
        # 简单添加CORS头
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response
    
    
    # 指定证书文件路径
    ssl_certfile = "cert.pem"
    ssl_keyfile = "key.pem"
    
    # 启动支持 HTTPS 的服务器
    print(f"以 HTTPS 模式启动服务器在 https://0.0.0.0:8889")
    app.include_router(prefix_router)
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=8889,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile
    )