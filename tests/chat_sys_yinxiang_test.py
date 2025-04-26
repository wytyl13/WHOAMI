import io
import pytest
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, Body, File, UploadFile, Request, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
import uvicorn
from llama_index.llms.ollama import Ollama
from typing import (
    Optional,
    Dict,
    List
)
import logging
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
import json
from pydantic import BaseModel
import httpx
import traceback
import aiohttp

from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.utils.R import R
from whoami.tool.llm_application.query_classifier import QueryClassifier
from whoami.tool.llm_application.chat_sys_test import ChatSys
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.llm_application.sx_coversation_history import SxConversationHistory
from whoami.utils.utils import Utils
from whoami.tool.llm_application.planning_agent import GoogleSearchTool
from whoami.tool.llm_application.planning_agent import HealthReportTool
from whoami.tool.llm_application.planning_agent import DirectLLMTool
from whoami.tool.agent.tool.direct_llm import DirectLLM
from whoami.tool.agent.tool.google_search import GoogleSearch
from whoami.tool.agent.tool.health_report import HealthReport
from whoami.tool.agent.tool import WeatherApi
from whoami.tool.agent.tool import Retrieval
from whoami.tool.agent.tool.planning_agent import PlanningAgent
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval

# llm_finetune = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))

enhance_qwen = EnhanceRetrieval(llm=llm_qwen)
retrieval = Retrieval()
direct_llm_tool = DirectLLM(enhance_llm=enhance_qwen)
google_search_tool = GoogleSearch(retrieval=retrieval)
health_report_tool = HealthReport(enhance_llm=enhance_qwen, device_sn='13D6F349200080712111957107')
weather_api = WeatherApi()
planning_agent = PlanningAgent(
    tools=[direct_llm_tool, google_search_tool, health_report_tool, weather_api], 
    enhance_llm=enhance_qwen
)

sql_config_path = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'
# 创建ChatSys实例，传入数据库提供者
sql_provider = SqlProvider(model=SxConversationHistory, sql_config_path=sql_config_path)
chat_sys = ChatSys(
    llm=llm_qwen, 
    sql_provider=sql_provider,
    planning_agent=planning_agent
)

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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置 API Key
API_KEY = "your_api_key_here"  # 更改为你的API key

# Dify API 请求模型
class DifyRequest(BaseModel):
    point: str
    params: dict = {}


class TTSRequest(BaseModel):
    text: str
    style: str = 'DEFAULT_STYLE'
    instruct: Optional[str] = None
    wait_complete: bool = False
    speed: float = 1.0
    use_batch: bool = False


# 请求数据模型
class RequestDataChatYinXiang(BaseModel):
    question: str
    user_id: Optional[str] = None
    messages: Optional[List] = None



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



class ConnectionManager:
    """管理WebSocket连接"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"客户端已连接: {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"客户端已断开连接: {user_id}")
    
    async def send_text(self, user_id: str, message: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)
    
    async def send_json(self, user_id: str, data: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(data)
    
    async def send_bytes(self, user_id: str, data: bytes):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_bytes(data)


manager = ConnectionManager()


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

stream_flag = 0

async def transcribe_audio(audio_data: bytes) -> str:
    """使用转录API转录音频数据"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {'audio_file': ('audio.pcm', io.BytesIO(audio_data), 'audio/pcm')}
        
        logger.info("发送数据到转录API")
        transcribe_response = await client.post(
            "http://1.71.15.121:8818/transcribe/test",
            files=files
        )
        
        if transcribe_response.status_code != 200:
            logger.error(f"转录API错误: {transcribe_response.status_code} - {transcribe_response.text}")
            raise Exception(f"转录失败: {transcribe_response.text}")
        
        transcribe_result = transcribe_response.json()
        transcribed_text = transcribe_result.get('text', '')
        if transcribe_result.get('status', '') == "failed":
            raise ValueError(transcribed_text)
            
            
        
            
        
        logger.info(f"转录结果: {transcribed_text}")
        return transcribed_text


async def stream_tts_audio(text: str, user_id: str):
    """
    调用外部ChatTTS API转换文本为语音，并实时流式返回
    """
    tts_api_url = "http://localhost:3000/tts/stream"  # 更新为您实际的TTS API URL
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info(f"发送文本到TTS API: {text[:50]}...")
        
        # 创建请求载荷
        payload = {
            "text": text,
            "pcm_flag": 1
        }
        
        # 调用TTS API并处理流式响应nohup pytest -s chat_sys_yinxiang_test.py > chat_sys_yinxiang_test_20250416.log 2>&1 &
        try:
            async with client.stream("POST", tts_api_url, json=payload) as response:
                if response.status_code != 200:
                    logger.error(f"TTS API错误: {response.status_code}")
                    response_text = await response.aread()
                    logger.error(f"错误详情: {response_text}")
                    raise Exception(f"TTS API请求失败，状态码: {response.status_code}")
                
                # 处理多部分响应
                boundary = None
                content_type = response.headers.get("content-type", "")
                if "boundary=" in content_type:
                    boundary = content_type.split("boundary=")[1]
                
                if not boundary:
                    logger.error("在多部分响应中未找到分界线")
                    raise Exception("无效的TTS API响应格式")
                
                logger.info(f"找到响应边界: {boundary}")
                
                # 用于解析多部分响应的变量
                buffer = bytearray()
                boundary_marker = f"--{boundary}".encode()
                end_marker = f"--{boundary}--".encode()
                is_final_message = False
                
                # 保存最后发送的文本，用于调试
                last_chunk_text = ""
                
                # 处理流式响应
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    
                    # 检查是否达到了结束标记
                    if end_marker in buffer:
                        is_final_message = True
                        logger.info("检测到流的结束标记")
                    
                    # 持续从缓冲区中提取完整部分
                    while True:
                        # 查找分界标记的位置
                        boundary_pos = buffer.find(boundary_marker)
                        if boundary_pos == -1:
                            break  # 没有找到更多部分
                        
                        # 检查这是否是最后一个边界标记
                        if buffer[boundary_pos:boundary_pos+len(end_marker)] == end_marker:
                            # 移除直到结束标记的所有内容
                            buffer = buffer[boundary_pos + len(end_marker):]
                            break
                        
                        # 查找下一个分界标记
                        next_boundary_pos = buffer.find(boundary_marker, boundary_pos + len(boundary_marker))
                        if next_boundary_pos == -1:
                            # 如果没有找到下一个边界，等待更多数据
                            if is_final_message:  # 除非这是最后一个消息
                                next_boundary_pos = len(buffer)
                            else:
                                break
                        
                        # 提取当前部分
                        part = buffer[boundary_pos:next_boundary_pos]
                        # 更新缓冲区，移除已处理的部分
                        buffer = buffer[next_boundary_pos:]
                        
                        # 处理这部分内容
                        if b"Content-Type: application/json" in part:
                            # 这是JSON元数据
                            try:
                                # 查找JSON开始的位置
                                json_start = part.find(b"\r\n\r\n") + 4
                                if json_start > 4:
                                    json_data = part[json_start:].decode('utf-8').strip()
                                    metadata = json.loads(json_data)
                                    
                                    # 记录元数据信息
                                    if "chunk_text" in metadata:
                                        last_chunk_text = metadata["chunk_text"]
                                        logger.info(f"收到元数据: 第 {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks', '?')} 部分, 文本: {last_chunk_text[:30]}...")
                                    else:
                                        logger.info(f"收到元数据: {metadata}")
                            except Exception as e:
                                logger.error(f"解析JSON元数据时出错: {str(e)}")
                        
                        elif b"Content-Type: audio/wav" in part:
                            # 这是音频数据
                            try:
                                # 查找音频数据开始的位置
                                audio_start = part.find(b"\r\n\r\n") + 4
                                if audio_start > 4:
                                    audio_data = part[audio_start:]
                                    
                                    # 仅发送非空的音频段
                                    if audio_data:
                                        logger.info(f"发送音频段，大小: {len(audio_data)} 字节，对应文本: {last_chunk_text[:30]}...")
                                        await manager.send_bytes(user_id, bytes(audio_data))
                                    else:
                                        logger.warning("收到空的音频段，跳过")
                            except Exception as e:
                                logger.error(f"处理音频数据时出错: {str(e)}")
                
                # 发送指示处理完成的消息
                await manager.send_json(user_id, {
                    "status": "completed",
                    "message": "TTS处理已完成"
                })
                
                logger.info("TTS流处理完成")
        
        except Exception as e:
            logger.error(f"处理TTS流时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 发送错误信息给客户端
            await manager.send_json(user_id, {
                "status": "error",
                "message": f"TTS处理失败: {str(e)}"
            })
            raise


def test_rag():
    
    @app.post('/chat_health_report')
    async def chat_health_report(request_data: RequestDataChat):
        # print(request_data)
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
                messages = result[0]["messages"][-6:] # 仅使用最后10条数据
            
        # 创建异步生成器以便与StreamingResponse一起使用
        if stream_flag:
            async def response_generator():
                async for text_chunk in chat_sys._run(
                    messages_history=messages, 
                    question=question, 
                    user_id=user_id, 
                    stream_flag=stream_flag
                ):
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
        else:
            # 对于非流式响应，收集单个完整响应
            response_content = ""
            response_generator = chat_sys._run(
                messages_history=messages, 
                question=question, 
                user_id=user_id, 
                stream_flag=stream_flag
            )
            async for chunk in response_generator:
                response_content = chunk
                break
            # 保存对话到数据库
            print("非流式响应完成，准备保存对话到数据库...")
            try:
                # 确保full_response已经设置，否则手动设置
                await chat_sys.save_qa_to_db(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=question
                )
                print(f"对话保存完成，response长度: {len(response_content)}")
            except Exception as e:
                print(f"保存对话失败: {str(e)}")
            
            # 返回JSON响应
            response = JSONResponse(
                content={"success": True, "data": response_content, "code": 200}
            )
            
        # 简单添加CORS头
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response


    @app.post('/get_conversation_history')
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


    @app.post('/truncate_conversation_history')
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
    
    
    @app.post('/chat_health_report_yinxiang_bake')
    async def chat_health_report_yinxiang_bake(
        file: UploadFile = File(...),
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = Form(None),
        messages: Optional[List] = None
    ):
        try:
            
            # 2. 调用语音转录API
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 准备文件数据
                files = {'audio_file': (file.filename, file.file, file.content_type)}
                
                # 调用语音转录API
                transcribe_response = await client.post(
                    "http://1.71.15.121:8818/transcribe/",
                    files=files
                )
                
                if transcribe_response.status_code != 200:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"语音转录失败: {transcribe_response.text}", "code": 500}
                    )
                
                # 获取转录结果
                transcribe_result = transcribe_response.json()
                # 假设转录API返回的文本在'text'字段中，根据实际情况调整
                transcribed_text = transcribe_result.get('text', '')
                
                if not transcribed_text:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "语音转录结果为空", "code": 400}
                    )
                
                print(f"语音转录结果: {transcribed_text}")
                
                # 3. 调用chat_health_report函数
                # 创建请求数据对象
                chat_request = RequestDataChat(
                    question=transcribed_text,
                    user_id=user_id,
                    messages=messages
                )
                print(chat_request.question)
                print(chat_request.user_id)
                print(chat_request.messages)
                # 直接调用现有的chat_health_report函数
                chat_response = await chat_health_report(chat_request)
                
                # 从chat_health_report的响应中提取文本内容
                if isinstance(chat_response, StreamingResponse):
                    # 如果是流式响应，需要收集所有内容
                    response_text = ""
                    async for chunk in chat_response.body_iterator:
                        if chunk.startswith(b'data: '):
                            response_text += chunk.decode('utf-8').replace('data: ', '')
                else:
                    # 如果chat_response已经是字典，直接使用
                    if isinstance(chat_response, dict):
                        response_dict = chat_response
                    # 如果是JSONResponse对象
                    elif hasattr(chat_response, 'json'):
                        response_dict = await chat_response.json()
                    # 如果是其他类型的响应对象
                    elif hasattr(chat_response, 'body'):
                        try:
                            response_dict = json.loads(chat_response.body)
                        except:
                            response_dict = {"data": str(chat_response.body)}
                    else:
                        # 如果都不是，创建一个默认的响应
                        response_dict = {"data": str(chat_response)}
                    
                    # 从response_dict中提取data字段
                    response_text = response_dict.get("data", "")
                
                # 4. 调用TTS API
                tts_request_data = TTSRequest(
                    text=response_text,
                    style="normal",  # 或其他风格
                    wait_complete=True,
                    use_batch=True
                )
                
                tts_response = await client.post(
                    "http://1.71.15.121:3000/tts",
                    json=tts_request_data.dict()
                )
                
                # 5. 直接返回TTS API的JSON响应
                if tts_response.status_code == 200:
                    return JSONResponse(content=tts_response.json())
                else:
                    return JSONResponse(
                        status_code=tts_response.status_code,
                        content={"success": False, "message": f"TTS API错误: {tts_response.text}", "code": tts_response.status_code}
                    )
                
        except Exception as e:
            import traceback
            print(f"处理过程中出错: {str(e)}")
            print(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"处理失败: {str(e)}", "code": 500}
            )
    
    
    
    @app.post('/chat_health_report_yinxiang')
    async def chat_health_report_yinxiang(
        request: Request,
        file: Optional[UploadFile] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = Form(None),
        messages: Optional[List] = None
    ):
        import io
        try:
            # 记录请求信息
            print("========== 新请求 ==========")
            print(f"Content-Type: {request.headers.get('content-type', 'None')}")
            print(f"请求方法: {request.method}")
            
            # 获取Query参数
            query_params = dict(request.query_params)
            if query_params:
                print(f"Query参数: {query_params}")
                # 如果URL中有user_id参数，使用它
                if "user_id" in query_params and not user_id:
                    user_id = query_params["user_id"]
                    print(f"从URL获取user_id: {user_id}")
            
            # 如果没有提供文件参数(FastAPI无法解析)，尝试从原始请求体获取
            if file is None:
                print("无法通过FastAPI解析文件，尝试从原始请求体获取")
                content_type = request.headers.get("content-type", "")
                
                # 如果是audio/pcm或其他音频类型直接获取请求体
                if "audio/" in content_type or content_type == "application/octet-stream":
                    body = await request.body()
                    print(f"收到原始音频数据: {len(body)} 字节")
                    
                    # 创建临时文件对象
                    file_like_object = io.BytesIO(body)
                    # 确保能够重新读取
                    file_like_object.seek(0)
                    
                    # 如果没有获取到user_id，使用默认值
                    if not user_id:
                        user_id = "13D6F349200080712111957107"  # 默认ID
                        print(f"使用默认user_id: {user_id}")
                else:
                    # 如果不是已知的音频类型，但也不是multipart/form-data
                    # 尝试解析表单数据
                    if "multipart/form-data" not in content_type:
                        print(f"未知的Content-Type: {content_type}，无法处理")
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "message": "无法识别的请求格式"}
                        )
            else:
                # FastAPI成功解析了文件，使用它
                print(f"通过FastAPI解析到文件: {file.filename}")
                file_content = await file.read()
                print(f"文件大小: {len(file_content)} 字节")
                
                # 创建新的文件对象并将指针重置
                file_like_object = io.BytesIO(file_content)
                await file.seek(0)  # 重置原始文件指针
            
            # 2. 调用语音转录API
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 准备文件数据 - 如果通过Form解析获取到了UploadFile对象
                if isinstance(file, UploadFile):
                    files = {'audio_file': (file.filename, file.file, file.content_type or 'audio/pcm')}
                else:
                    # 如果是通过请求体直接获取的音频数据
                    files = {'audio_file': ('audio.pcm', file_like_object, 'audio/pcm')}
                
                print("发送数据到转录API")
                # 调用语音转录API
                transcribe_response = await client.post(
                    "http://1.71.15.121:8818/transcribe/",
                    files=files
                )
                
                if transcribe_response.status_code != 200:
                    print(f"转录API错误: {transcribe_response.status_code} - {transcribe_response.text}")
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"语音转录失败: {transcribe_response.text}", "code": 500}
                    )
                
                # 获取转录结果
                transcribe_result = transcribe_response.json()
                # 假设转录API返回的文本在'text'字段中，根据实际情况调整
                transcribed_text = transcribe_result.get('text', '')
                
                print(f"转录结果: {transcribed_text}")
                
                if not transcribed_text:
                    print("转录结果为空")
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "语音转录结果为空", "code": 400}
                    )
                
                # 3. 调用chat_health_report函数
                # 创建请求数据对象
                chat_request = RequestDataChat(
                    question=transcribed_text,
                    user_id=user_id,
                    messages=messages
                )
                print(f"问题: {chat_request.question}")
                print(f"用户ID: {chat_request.user_id}")
                
                # 直接调用现有的chat_health_report函数
                chat_response = await chat_health_report(chat_request)
                
                # 从chat_health_report的响应中提取文本内容
                if isinstance(chat_response, StreamingResponse):
                    # 如果是流式响应，需要收集所有内容
                    response_text = ""
                    async for chunk in chat_response.body_iterator:
                        if chunk.startswith(b'data: '):
                            response_text += chunk.decode('utf-8').replace('data: ', '')
                else:
                    # 如果chat_response已经是字典，直接使用
                    if isinstance(chat_response, dict):
                        response_dict = chat_response
                    # 如果是JSONResponse对象
                    elif hasattr(chat_response, 'json'):
                        response_dict = await chat_response.json()
                    # 如果是其他类型的响应对象
                    elif hasattr(chat_response, 'body'):
                        try:
                            response_dict = json.loads(chat_response.body)
                        except:
                            response_dict = {"data": str(chat_response.body)}
                    else:
                        # 如果都不是，创建一个默认的响应
                        response_dict = {"data": str(chat_response)}
                    
                    # 从response_dict中提取data字段
                    response_text = response_dict.get("data", "")
                
                print(f"聊天响应: {response_text[:100]}...")  # 只记录前100个字符
                
                # 4. 调用TTS API
                tts_request_data = TTSRequest(
                    text=response_text,
                    style="normal",  # 或其他风格
                    wait_complete=True,
                    use_batch=True
                )
                
                print("发送请求到TTS API")
                tts_response = await client.post(
                    "http://1.71.15.121:3000/tts",
                    json=tts_request_data.dict()
                )
                
                # 5. 获取TTS API返回的audio_url并下载音频
                if tts_response.status_code == 200:
                    tts_result = tts_response.json()
                    audio_url = tts_result.get('audio_url', '')
                    print(f"TTS成功，获取到音频URL: {audio_url}")
                    
                    if not audio_url:
                        return JSONResponse(
                            status_code=500,
                            content={"success": False, "message": "TTS API未返回音频URL", "code": 500}
                        )
                    
                    # 下载音频文件
                    print(f"正在下载音频文件: {audio_url}")
                    audio_response = await client.get(audio_url)
                    
                    if audio_response.status_code != 200:
                        print(f"下载音频失败: {audio_response.status_code} - {audio_response.text}")
                        return JSONResponse(
                            status_code=500,
                            content={"success": False, "message": f"下载音频失败: {audio_response.text}", "code": 500}
                        )
                        
                    # 获取音频数据
                    audio_data = audio_response.content
                    
                    # 使用pydub转换为PCM格式，采样率8000
                    try:
                        # 需要导入这些库
                        from pydub import AudioSegment
                        import io
                        
                        # 从二进制数据创建音频对象
                        # 如果不确定格式，可以根据URL后缀或Content-Type判断
                        format_hint = audio_url.split('.')[-1].lower() if '.' in audio_url else 'mp3'
                        # 如果响应头中有Content-Type，也可以用它来判断格式
                        content_type = audio_response.headers.get('content-type', '')
                        if 'wav' in content_type:
                            format_hint = 'wav'
                        elif 'mpeg' in content_type or 'mp3' in content_type:
                            format_hint = 'mp3'
                            
                        print(f"检测到音频格式: {format_hint}")
                        
                        # 从二进制数据加载音频
                        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format_hint)
                        
                        # 转换为PCM格式，采样率8000
                        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                        
                        # 导出为PCM格式
                        pcm_buffer = io.BytesIO()
                        audio.export(pcm_buffer, format="s16le")  # raw PCM格式
                        pcm_data = pcm_buffer.getvalue()
                        
                        print(f"成功转换为PCM格式，大小: {len(pcm_data)} 字节")
                        
                        # 返回PCM音频数据，注意设置正确的Content-Type
                        return Response(
                            content=pcm_data,
                            media_type="audio/pcm",
                            headers={
                                "Content-Disposition": "attachment; filename=response.pcm"
                            }
                        )
                        
                    except Exception as e:
                        print(f"音频转换失败: {str(e)}")
                        return JSONResponse(
                            status_code=500,
                            content={"success": False, "message": f"音频转换失败: {str(e)}", "code": 500}
                        )
                else:
                    print(f"TTS API错误: {tts_response.status_code} - {tts_response.text}")
                    return JSONResponse(
                        status_code=tts_response.status_code,
                        content={"success": False, "message": f"TTS API错误: {tts_response.text}", "code": tts_response.status_code}
                    )
                
        except Exception as e:
            import traceback
            print(f"处理过程中出错: {str(e)}")
            print(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"处理失败: {str(e)}", "code": 500}
            )
    
    
    
    @app.post('/chat_health_report_yinxiang_')
    async def chat_health_report_yinxiang_(
        request: Request,
        file: Optional[UploadFile] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = Form(None),
        messages: Optional[List] = None
    ):
        try:
            # 记录请求信息
            print("========== 新请求 ==========")
            print(f"Content-Type: {request.headers.get('content-type', 'None')}")
            print(f"请求方法: {request.method}")
            
            # 获取Query参数
            query_params = dict(request.query_params)
            if query_params:
                print(f"Query参数: {query_params}")
                # 如果URL中有user_id参数，使用它
                if "user_id" in query_params and not user_id:
                    user_id = query_params["user_id"]
                    print(f"从URL获取user_id: {user_id}")
            
            # 如果没有提供文件参数(FastAPI无法解析)，尝试从原始请求体获取
            if file is None:
                print("无法通过FastAPI解析文件，尝试从原始请求体获取")
                content_type = request.headers.get("content-type", "")
                
                # 如果是audio/pcm或其他音频类型直接获取请求体
                if "audio/" in content_type or content_type == "application/octet-stream":
                    body = await request.body()
                    print(f"收到原始音频数据: {len(body)} 字节")
                    
                    # 创建临时文件对象
                    file_like_object = io.BytesIO(body)
                    # 确保能够重新读取
                    file_like_object.seek(0)
                    
                    # 如果没有获取到user_id，使用默认值
                    if not user_id:
                        user_id = "13D6F349200080712111957107"  # 默认ID
                        print(f"使用默认user_id: {user_id}")
                else:
                    # 如果不是已知的音频类型，但也不是multipart/form-data
                    # 尝试解析表单数据
                    if "multipart/form-data" not in content_type:
                        print(f"未知的Content-Type: {content_type}，无法处理")
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "message": "无法识别的请求格式"}
                        )
            else:
                # FastAPI成功解析了文件，使用它
                print(f"通过FastAPI解析到文件: {file.filename}")
                file_content = await file.read()
                print(f"文件大小: {len(file_content)} 字节")
                
                # 创建新的文件对象并将指针重置
                file_like_object = io.BytesIO(file_content)
                await file.seek(0)  # 重置原始文件指针
            
            # 2. 调用语音转录API
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 准备文件数据 - 如果通过Form解析获取到了UploadFile对象
                if isinstance(file, UploadFile):
                    files = {'audio_file': (file.filename, file.file, file.content_type or 'audio/pcm')}
                else:
                    # 如果是通过请求体直接获取的音频数据
                    files = {'audio_file': ('audio.pcm', file_like_object, 'audio/pcm')}
                
                print("发送数据到转录API")
                # 调用语音转录API
                transcribe_response = await client.post(
                    "http://1.71.15.121:8818/transcribe/",
                    files=files
                )
                
                if transcribe_response.status_code != 200:
                    print(f"转录API错误: {transcribe_response.status_code} - {transcribe_response.text}")
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"语音转录失败: {transcribe_response.text}", "code": 500}
                    )
                
                # 获取转录结果
                transcribe_result = transcribe_response.json()
                # 假设转录API返回的文本在'text'字段中，根据实际情况调整
                transcribed_text = transcribe_result.get('text', '')
                
                print(f"转录结果: {transcribed_text}")
                
                if not transcribed_text:
                    print("转录结果为空")
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "语音转录结果为空", "code": 400}
                    )
                
                # 3. 调用chat_health_report函数
                # 创建请求数据对象
                chat_request = RequestDataChat(
                    question=transcribed_text,
                    user_id=user_id,
                    messages=messages
                )
                print(f"问题: {chat_request.question}")
                print(f"用户ID: {chat_request.user_id}")
                
                # 直接调用现有的chat_health_report函数
                chat_response = await chat_health_report(chat_request)
                
                # 从chat_health_report的响应中提取文本内容
                if isinstance(chat_response, StreamingResponse):
                    # 如果是流式响应，需要收集所有内容
                    response_text = ""
                    async for chunk in chat_response.body_iterator:
                        if chunk.startswith(b'data: '):
                            response_text += chunk.decode('utf-8').replace('data: ', '')
                else:
                    # 如果chat_response已经是字典，直接使用
                    if isinstance(chat_response, dict):
                        response_dict = chat_response
                    # 如果是JSONResponse对象
                    elif hasattr(chat_response, 'json'):
                        response_dict = await chat_response.json()
                    # 如果是其他类型的响应对象
                    elif hasattr(chat_response, 'body'):
                        try:
                            response_dict = json.loads(chat_response.body)
                        except:
                            response_dict = {"data": str(chat_response.body)}
                    else:
                        # 如果都不是，创建一个默认的响应
                        response_dict = {"data": str(chat_response)}
                    
                    # 从response_dict中提取data字段
                    response_text = response_dict.get("data", "")
                
                print(f"聊天响应: {response_text[:100]}...")  # 只记录前100个字符
                
                # 4. 调用TTS API
                tts_request_data = TTSRequest(
                    text=response_text,
                    style="normal",  # 或其他风格
                    wait_complete=True,
                    use_batch=True
                )
                
                print("发送请求到TTS API")
                tts_response = await client.post(
                    "http://1.71.15.121:3000/tts",
                    json=tts_request_data.dict()
                )
                
                # 5. 直接返回TTS API的JSON响应
                if tts_response.status_code == 200:
                    tts_result = tts_response.json()
                    print(f"TTS成功: {tts_result.get('audio_url', '')}")
                    return JSONResponse(content=tts_result)
                else:
                    print(f"TTS API错误: {tts_response.status_code} - {tts_response.text}")
                    return JSONResponse(
                        status_code=tts_response.status_code,
                        content={"success": False, "message": f"TTS API错误: {tts_response.text}", "code": tts_response.status_code}
                    )
                
        except Exception as e:
            import traceback
            print(f"处理过程中出错: {str(e)}")
            print(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"处理失败: {str(e)}", "code": 500}
            )
    
    
    
    @app.websocket("/ws/health_report/{user_id}")
    async def websocket_health_report(websocket: WebSocket, user_id: str):
        """健康报告服务的WebSocket端点 - 仅音频"""
        await manager.connect(websocket, user_id)
        
        try:
            while True:
                # 等待来自客户端的数据 - 仅期望音频数据
                data = await websocket.receive()
                
                # 仅处理音频数据
                if "bytes" in data:
                    # 处理音频数据
                    audio_data = data["bytes"]
                    logger.info(f"从 {user_id} 接收到音频数据: {len(audio_data)} 字节")
                    
                    try:
                        # 1. 转录音频
                        transcribed_text = await transcribe_audio(audio_data)
                        if transcribed_text == "":
                            # 向客户端发送转录为空的通知
                            await manager.send_json(user_id, {
                                "type": "transcription_empty",
                                "message": "未能识别出您的语音，请再试一次。"
                            })
                            
                            # 发送完成信号，告知客户端这个会话已经处理完成
                            await manager.send_json(user_id, {
                                "status": "completed",
                                "message": "TTS处理已完成"
                            })
                            
                            # 继续下一次循环，等待新的音频输入
                            continue
                        # 2. 向客户端发送转录结果
                        await manager.send_json(user_id, {
                            "type": "transcription",
                            "text": transcribed_text
                        })
                        
                        # 3. 使用转录文本直接调用chat_health_report
                        chat_request = RequestDataChat(
                            question=transcribed_text,
                            user_id=user_id,
                            messages=None,
                            conversation_id=None
                        )
                        
                        logger.info(f"为 {user_id} 调用chat_health_report: {transcribed_text}")
                        # 直接函数调用
                        chat_response = await chat_health_report(chat_request)
                        
                        # 4. 从chat_health_report响应中提取文本
                        # 从chat_health_report的响应中提取文本内容
                        if isinstance(chat_response, StreamingResponse):
                            # 如果是流式响应，需要收集所有内容
                            response_text = ""
                            async for chunk in chat_response.body_iterator:
                                if chunk.startswith(b'data: '):
                                    response_text += chunk.decode('utf-8').replace('data: ', '')
                        else:
                            # 如果chat_response已经是字典，直接使用
                            if isinstance(chat_response, dict):
                                response_dict = chat_response
                            # 如果是JSONResponse对象
                            elif hasattr(chat_response, 'json'):
                                response_dict = await chat_response.json()
                            # 如果是其他类型的响应对象
                            elif hasattr(chat_response, 'body'):
                                try:
                                    response_dict = json.loads(chat_response.body)
                                except:
                                    response_dict = {"data": str(chat_response.body)}
                            else:
                                # 如果都不是，创建一个默认的响应
                                response_dict = {"data": str(chat_response)}
                            
                            # 从response_dict中提取data字段
                            response_text = response_dict.get("data", "")
                        
                        print(f"聊天响应: {response_text[:100]}...")  # 只记录前100个字符
                        
                        # 5. 向客户端发送聊天响应
                        await manager.send_json(user_id, {
                            "type": "chat_response",
                            "text": response_text
                        })
                        
                        # 6. 开始流式发送TTS音频
                        await manager.send_json(user_id, {
                            "type": "audio_start"
                        })
                        
                        # 调用TTS API并实时流式返回音频段
                        await stream_tts_audio(response_text, user_id)
                        
                        # 发送音频结束信号
                        await manager.send_json(user_id, {
                            "type": "audio_end"
                        })
                        
                    except Exception as e:
                        logger.error(f"处理请求时出错: {str(e)}")
                        logger.error(traceback.format_exc())
                        await manager.send_json(user_id, {
                            "type": "error",
                            "message": f"处理失败: {str(e)}"
                        })
                else:
                    # 通知客户端仅支持音频数据
                    await manager.send_json(user_id, {
                        "type": "error",
                        "message": "此端点仅支持音频数据"
                    })
        
        except WebSocketDisconnect:
            manager.disconnect(user_id)
        except Exception as e:
            logger.error(f"WebSocket错误: {str(e)}")
            logger.error(traceback.format_exc())
            manager.disconnect(user_id)
    
    
    
    
    async def get_complete_tts_audio(text: str) -> bytes:
        """
        将文本转换为语音，并返回完整的音频数据
        
        Args:
            text: 要转换为语音的文本
        
        Returns:
            bytes: 完整的音频数据
        """
        try:
            # 初始化一个BytesIO对象用于收集所有音频数据
            audio_segments = bytearray()
            
            # 设置TTS服务的URL
            tts_url = "http://localhost:3000/tts_url"  # 根据实际TTS服务地址调整
            
            # 发送请求到TTS服务
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    tts_url,
                    json={"text": text, "pcm_flag": 0},  # 使用WAV格式
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"TTS服务返回错误: {response.status} - {error_text}")
                        return b""
                    
                    # 读取流式响应并拼接
                    logger.info("开始接收TTS音频流...")
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            audio_segments.extend(chunk)
            
            logger.info(f"TTS音频接收完成，总大小: {len(audio_segments)} 字节")
            return bytes(audio_segments)
            
        except aiohttp.ClientError as e:
            logger.error(f"TTS服务连接错误: {str(e)}")
            logger.error(traceback.format_exc())
            return b""
        except Exception as e:
            logger.error(f"TTS处理时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return b""
    
    
    
    
    @app.post("/api/health_report/{user_id}", response_class=Response)
    async def http_health_report(request: Request, user_id: str):
        """健康报告服务的HTTP端点 - 仅音频输入，返回完整音频"""
        try:
            # 读取请求体中的音频数据
            audio_data = await request.body()
            logger.info(f"从 {user_id} 接收到音频数据: {len(audio_data)} 字节")
            
            # 1. 转录音频
            transcribed_text = await transcribe_audio(audio_data)
            if transcribed_text == "":
                # 如果转录为空，返回错误响应
                return JSONResponse(
                    status_code=400,
                    content={"error": "未能识别出您的语音，请再试一次。"}
                )
            
            # 2. 使用转录文本调用chat_health_report
            chat_request = RequestDataChat(
                question=transcribed_text,
                user_id=user_id,
                messages=None,
                conversation_id=None
            )
            
            logger.info(f"为 {user_id} 调用chat_health_report: {transcribed_text}")
            chat_response = await chat_health_report(chat_request)
            
            # 3. 从chat_health_report响应中提取文本
            if isinstance(chat_response, StreamingResponse):
                # 如果是流式响应，需要收集所有内容
                response_text = ""
                async for chunk in chat_response.body_iterator:
                    if chunk.startswith(b'data: '):
                        response_text += chunk.decode('utf-8').replace('data: ', '')
            else:
                # 如果chat_response已经是字典，直接使用
                if isinstance(chat_response, dict):
                    response_dict = chat_response
                # 如果是JSONResponse对象
                elif hasattr(chat_response, 'json'):
                    response_dict = await chat_response.json()
                # 如果是其他类型的响应对象
                elif hasattr(chat_response, 'body'):
                    try:
                        response_dict = json.loads(chat_response.body)
                    except:
                        response_dict = {"data": str(chat_response.body)}
                else:
                    # 如果都不是，创建一个默认的响应
                    response_dict = {"data": str(chat_response)}
                
                # 从response_dict中提取data字段
                response_text = response_dict.get("data", "")
            
            logger.info(f"聊天响应: {response_text[:100]}...")  # 只记录前100个字符
            
            # 4. 调用TTS API并收集所有音频数据
            # 设置TTS服务的URL
            tts_url = "http://localhost:3000/tts_url"  # 根据实际TTS服务地址调整

            # 发送请求到TTS服务
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    tts_url,
                    json={"text": response_text},  # 使用WAV格式
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"TTS服务返回错误: {response.status} - {error_text}")
                        return JSONResponse(
                            status_code=500,
                            content={"error": f"TTS服务返回错误: {response.status}"}
                        )
                    
                    # 解析JSON响应，获取URL
                    response_data = await response.json()
                    audio_url = response_data.get("url", "")
                    file_path = response_data.get("file_path", "")
                    
                    logger.info(f"TTS音频已生成，URL: {audio_url}, 文件路径: {file_path}")
        
                    # 返回成功响应
                    return JSONResponse(
                        status_code=200,
                        content={"url": audio_url, "file_path": file_path}
                    )
            
            
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"处理失败: {str(e)}"}
            )
    
    
    
    
    @app.post("/api/health_report/text/{user_id}", response_class=StreamingResponse)
    async def http_health_report_text(request: Request, user_id: str):
        """健康报告服务的HTTP端点 - 流式返回转录文本、AI回复文本和音频URL"""
        # 读取请求体中的音频数据
        audio_data = await request.body()
        logger.info(f"从 {user_id} 接收到音频数据: {len(audio_data)} 字节")
        
        async def stream_response(audio_data):
            try:
                
                # 1. 转录音频
                try:
                    transcribed_text = await transcribe_audio(audio_data)
                    if transcribed_text == "":
                        # 如果转录为空，返回错误响应
                        error_json = json.dumps({"type": "error", "data": "未能识别出您的语音，请再试一次。"})
                        yield f"data: {error_json}\n\n"
                        return
                except Exception as e:
                    error_json = json.dumps({"type": "error", "data": "str(e)"})
                    yield f"data: {error_json}\n\n"
                    return
                
                # 返回转录文本结果
                user_json = json.dumps({"type": "user", "data": transcribed_text})
                logger.info(f"data: {user_json}\n\n")
                yield f"data: {user_json}\n\n"
                
                # 2. 使用转录文本调用chat_health_report
                chat_request = RequestDataChat(
                    question=transcribed_text,
                    user_id=user_id,
                    messages=None,
                    conversation_id=None
                )
                
                logger.info(f"为 {user_id} 调用chat_health_report: {transcribed_text}")
                chat_response = await chat_health_report(chat_request)
                
                # 3. 从chat_health_report响应中提取文本
                response_text = ""
                if isinstance(chat_response, StreamingResponse):
                    # 如果是流式响应，需要收集所有内容
                    async for chunk in chat_response.body_iterator:
                        if chunk.startswith(b'data: '):
                            chunk_text = chunk.decode('utf-8').replace('data: ', '')
                            response_text += chunk_text
                            
                            # 将AI回复实时传送给客户端
                            ai_response_json = json.dumps({"type": "ai", "data": chunk_text})
                            logger.info(f"data: {user_json}\n\n")
                            yield f"data: {ai_response_json}\n\n"
                else:
                    # 如果chat_response已经是字典，直接使用
                    if isinstance(chat_response, dict):
                        response_dict = chat_response
                    # 如果是JSONResponse对象
                    elif hasattr(chat_response, 'json'):
                        response_dict = await chat_response.json()
                    # 如果是其他类型的响应对象
                    elif hasattr(chat_response, 'body'):
                        try:
                            response_dict = json.loads(chat_response.body)
                        except:
                            response_dict = {"data": str(chat_response.body)}
                    else:
                        # 如果都不是，创建一个默认的响应
                        response_dict = {"data": str(chat_response)}
                    
                    # 从response_dict中提取data字段
                    response_text = response_dict.get("data", "")
                    
                    # 返回AI回复文本结果
                    ai_response_json = json.dumps({"type": "ai", "data": response_text})
                    logger.info(f"data: {user_json}\n\n")
                    yield f"data: {ai_response_json}\n\n"
                
                logger.info(f"聊天响应: {response_text[:100]}...")  # 只记录前100个字符
                
                # 4. 调用TTS API并收集所有音频数据
                # 设置TTS服务的URL
                tts_url = "http://localhost:3000/tts_url"  # 根据实际TTS服务地址调整

                # 发送请求到TTS服务
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        tts_url,
                        json={"text": response_text},
                        timeout=60
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"TTS服务返回错误: {response.status} - {error_text}")
                            error_json = json.dumps({"type": "error", "data": f"TTS服务返回错误: {response.status}"})
                            yield f"data: {error_json}\n\n"
                            return
                        
                        # 解析JSON响应，获取URL和文件路径
                        response_data = await response.json()
                        audio_url = response_data.get("url", "")
                        file_path = response_data.get("file_path", "")
                        
                        logger.info(f"TTS音频已生成，URL: {audio_url}, 文件路径: {file_path}")
                
                        # 返回音频URL结果
                        audio_json = json.dumps({"type": "audio_url", "data": audio_url})
                        logger.info(f"data: {audio_json}\n\n")
                        yield f"data: {audio_json}\n\n"
                
            except Exception as e:
                logger.error(f"处理请求时出错: {str(e)}")
                logger.error(traceback.format_exc())
                error_json = json.dumps({"type": "error", "data": f"处理失败: {str(e)}"})
                logger.info(f"data: {error_json}\n\n")
                yield f"data: {error_json}\n\n"
        
        return StreamingResponse(
            stream_response(audio_data),
            media_type="text/event-stream"
        )
    
    
    
    
    # 启动支持 HTTPS 的服务器
    print(f"以 HTTPS 模式启动服务器在 https://0.0.0.0:8888")
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=8888,
    )