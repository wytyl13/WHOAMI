from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Optional
import sys
import os
from datetime import datetime
import logging
from fastapi.encoders import jsonable_encoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入相关模块
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.streamlit.health_report.table.community_real_time_data import CommunityRealTimeData
from whoami.tool.streamlit.health_report.table.user_data import UserData
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.tool.real_time_vital_analyze.sleep_statistics_model import SleepStatistics
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval
from pathlib import Path
from whoami.tool.agent.tool.direct_llm_community_ai_user import DirectLLMCommunityAiUser

llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))

enhance_qwen_user = EnhanceRetrieval(llm=llm_qwen, data_dir="/work/ai/WHOAMI/retrieval_data", index_dir="/work/ai/WHOAMI/retrieval_storage")
direct_llm_tool = DirectLLMCommunityAiUser(enhance_llm=enhance_qwen_user)


# 配置文件路径
SQL_CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'

# 定义请求模型
class DeviceInfo(BaseModel):
    device_code: str
    wifi_name: str
    wifi_password: str
    working_distance: Optional[float] = 1.5
    scene: Optional[str] = "睡眠监测"
    username: Optional[str] = ""
    user_id: Optional[int] = None


class CommunityRealTimeInfo(BaseModel):
    type: str
    content: str
    username: Optional[str] = ""
class ChatInfo(BaseModel):
    question: Optional[str] = None
    
class ListCommunityRealTimeInfo(BaseModel):
    type: Optional[str] = None
    username: Optional[str] = None
class ListSleepStatistics(BaseModel):
    username: Optional[str] = None

app = FastAPI(title="AeroSense设备配网API", version="1.0.0")

# 更详细的CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8000",
        "https://127.0.0.1:8000", 
        "https://1.71.15.121:8000",  # 添加你的外网IP
        "https://ai.shunxikj.com:8000",  # 添加你的外网IP
        "https://localhost:8889",  
        "https://localhost:8890",  
        "https://127.0.0.1:8889",
        "https://127.0.0.1:8890",
        "https://1.71.15.121:8889",  # 添加你的外网IP
        "https://1.71.15.121:8890",  # 添加你的外网IP
        "https://ai.shunxikj.com:8889",  # 添加你的外网IP
        "https://ai.shunxikj.com:8890",  # 添加你的外网IP
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8000",
#         "http://127.0.0.1:8000", 
#         "http://1.71.15.121:8000",  # 添加你的外网IP
#         "http://localhost:8889",  
#         "http://127.0.0.1:8889",
#         "http://1.71.15.121:8889",  # 添加你的外网IP
#     ],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],
# )



@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[收到请求] {request.method} {request.url}")
    if request.method == "POST":
        body = await request.body()
        print(f"[POST BODY] {body.decode()}")
    response = await call_next(request)
    print(f"[响应状态] {response.status_code}")
    return response


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API服务运行正常", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    return {"message": "AeroSense设备配网API服务", "version": "1.0.0"}

@app.get("/docs")
async def custom_docs():
    """自定义文档页面，确保可访问"""
    return {"message": "API文档可用", "docs_url": "/docs"}

@app.post("/api/save_device")
async def save_device(device_info: DeviceInfo):
    try:
        logger.info(f"收到设备保存请求: {device_info.device_code}")
        
        # 验证必要字段
        if not device_info.device_code:
            raise HTTPException(status_code=400, detail="设备编号不能为空")
        if not device_info.wifi_name:
            raise HTTPException(status_code=400, detail="WiFi名称不能为空")
        if not device_info.wifi_password:
            raise HTTPException(status_code=400, detail="WiFi密码不能为空")
        
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        
        # 检查现有设备
        existing_devices = sql_provider.get_record_by_condition(
            condition={"device_code": device_info.device_code}
        )
        
        # 软删除现有设备
        if existing_devices:
            logger.info(f"找到 {len(existing_devices)} 个现有设备，进行软删除")
            for device in existing_devices:
                sql_provider.delete_record(
                    record_id=device["id"],
                )
        
        # 准备新设备数据
        device_data = {
            "device_code": device_info.device_code,
            "scene": device_info.scene,
            "wifi_name": device_info.wifi_name,
            "wifi_password": device_info.wifi_password,
            "username": device_info.username or "unknown",
            "user_id": device_info.user_id or 1,
            "status": "active",
            "creator": device_info.username or "system",
            "updater": device_info.username or "system",
            "deleted": False,
            "tenant_id": 0,
            "create_time": datetime.now(),
            "update_time": datetime.now()
        }
        
        logger.info(f"准备插入设备数据: {device_data}")
        
        # 执行数据库插入
        result = sql_provider.add_record(device_data)
        
        if result:
            logger.info(f"设备保存成功: {device_info.device_code}")
            response_data = {
                "success": True, 
                "message": "设备信息已成功保存", 
                "device_code": device_info.device_code,
                "timestamp": datetime.now().isoformat()
            }
            print(f"[SUCCESS] 准备返回响应: {response_data}")  # 添加这行
            return JSONResponse(
                content=response_data,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        else:
            logger.error(f"数据库添加记录失败: {device_info.device_code}")
            raise HTTPException(status_code=500, detail="数据库添加记录失败")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] 异常详情: {str(e)}")  # 添加这行
        print(f"[ERROR] 异常类型: {type(e)}")   # 添加这行
        logger.error(f"保存设备信息异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存设备信息失败: {str(e)}")




@app.post("/api/save_community_real_time_data")
async def save_community_real_time_data(community_real_time_info: CommunityRealTimeInfo):
    try:
        logger.info(f"收到数据保存请求: {community_real_time_info}")
        
        # 验证必要字段
        if not community_real_time_info.type:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "请提供内容类型（时讯消息/通告）", "data": None, "timestamp": datetime.now().isoformat()}
            )

        if not community_real_time_info.content:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"收到发布{community_real_time_info.type}，请提供具体内容！", "data": None, "timestamp": datetime.now().isoformat()}
            )

        try:
            sql_provider = SqlProvider(model=CommunityRealTimeData, sql_config_path=SQL_CONFIG_PATH)
            # 准备新设备数据
            insert_data = {
                "type": community_real_time_info.type,
                "content": community_real_time_info.content,
                "creator": community_real_time_info.username,
                "updater": community_real_time_info.username,
                "create_time": datetime.now(),
                "update_time": datetime.now()
            }
            logger.info(f"准备插入数据: {insert_data}")
            # 执行数据库插入
            result = sql_provider.add_record(insert_data)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": True, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
            
        if result:
            logger.info(f"数据保存成功: {community_real_time_info}")
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "成功！", "data": str(result), "timestamp": datetime.now().isoformat()}
            )
        else:
            logger.error(f"数据库添加记录失败: {community_real_time_info}")
            return JSONResponse(
                status_code=500,
                content={"success": True, "message": f"数据库添加记录失败: {community_real_time_info}", "data": None, "timestamp": datetime.now().isoformat()}
            )
    except Exception as e:
        logger.error(f"数据库添加记录失败: {community_real_time_info}")
        return JSONResponse(
            status_code=500,
            content={"success": True, "message": f"数据库添加记录失败: {community_real_time_info}", "data": None, "timestamp": datetime.now().isoformat()}
        )


@app.post("/api/list_community_real_time_data")
async def save_community_real_time_data(list_community_real_time_info: ListCommunityRealTimeInfo):
    if list_community_real_time_info.username is None or list_community_real_time_info.username == "":
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "用户名不能为空！", "data": None, "timestamp": datetime.now().isoformat()}
        )
    
    try:
        sql_provider = SqlProvider(model=CommunityRealTimeData, sql_config_path=SQL_CONFIG_PATH)
        logger.info(list_community_real_time_info)
        result = sql_provider.get_record_by_condition(
            condition={"creator": list_community_real_time_info.username} if list_community_real_time_info.type is None or list_community_real_time_info.type == "" else {"creator": list_community_real_time_info.username, "type": list_community_real_time_info.type},
            fields=["id", "type", "content", "create_time"]
        )
        result = result[-5:] if len(result) > 5 else result
        json_compatible_result = jsonable_encoder(result)
        logger.info(json_compatible_result)
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "成功！", "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
        )


@app.get("/api/list_all_user_data")
async def list_all_user_data():
    try:
        sql_provider = SqlProvider(model=UserData, sql_config_path=SQL_CONFIG_PATH)
        result = sql_provider.get_record_by_condition(
            fields=["username", "full_name", "address"]
        )
        json_compatible_result = jsonable_encoder(result)
        logger.info(json_compatible_result)
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "成功！", "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
        )


@app.post("/api/list_sleep_statistics")
async def list_sleep_statistics(list_sleep_statistics_: ListSleepStatistics):
    if list_sleep_statistics_.username is None or list_sleep_statistics_.username == "":
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "用户名不能为空！", "data": None, "timestamp": datetime.now().isoformat()}
        )
    try:
        sql_provider_device_data = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        device_data_result = sql_provider_device_data.get_record_by_condition(
            condition={"username": list_sleep_statistics_.username},
            fields=["device_code"]
        )
        if not device_data_result:
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "成功", "data": [], "timestamp": datetime.now().isoformat()}
            )
        
        sql_provider_sleep_statistics = SqlProvider(model=SleepStatistics, sql_config_path=SQL_CONFIG_PATH)
        result = sql_provider_sleep_statistics.get_record_by_condition(
            condition={"device_sn": device_data_result[0]["device_code"]},
            fields=[
                "sleep_start_time", "sleep_end_time", "health_report"
            ]
        )
        json_compatible_result = jsonable_encoder(result)
        logger.info(json_compatible_result)
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "成功！", "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
        )
    
    
    
    

# 添加健康检查和测试端点
@app.get("/api/test")
async def test_endpoint():
    """测试端点，验证API可达性"""
    return {
        "status": "success",
        "message": "API测试成功",
        "timestamp": datetime.now().isoformat(),
        "server_info": {
            "host": "0.0.0.0",
            "port": 8889
        }
    }


@app.post("/api/test_save")
async def test_save():
    """测试保存端点"""
    test_device = DeviceInfo(
        device_code="TEST_DEVICE_001",
        wifi_name="TEST_WIFI",
        wifi_password="12345678",
        username="test_user"
    )
    
    return await save_device(test_device)



@app.post("/api/chat")
async def save_community_real_time_data(chat_info: ChatInfo):
    try:
        logger.info(f"收到数据保存请求: {chat_info}")
        
        # 验证必要字段
        if not chat_info.question:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "请提供具体问题！", "data": None, "timestamp": datetime.now().isoformat()}
            )

        try:
            params = {
                "question": chat_info.question,
                "message_history": None,
                "username": "temp",
                "location": "none",
                "role": "none"
            }
            response = ""
            async for chunk in direct_llm_tool.execute(
                **params
            ):
                response += chunk
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": f"操作成功！", "data": response, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
        )
            


if __name__ == "__main__":
    print("🚀 启动AeroSense设备配网API服务...")
    print("📡 服务地址: http://localhost:8889")
    print("📋 API文档: http://localhost:8889/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8889,
        log_level="info",
        reload=False,
        ssl_certfile="/work/ai/WHOAMI/tests/shunxikj.com.crt",  # 添加这行
        ssl_keyfile="/work/ai/WHOAMI/tests/shunxikj.com.key"     # 添加这行
    )