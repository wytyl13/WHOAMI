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

app = FastAPI(title="AeroSense设备配网API", version="1.0.0")

# HTTP版本的CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000", 
        "http://localhost:8889",  
        "http://127.0.0.1:8889",
        "https://ai.shunxikj.com",  # 允许nginx代理访问
        "*"  # 开发阶段可以使用，生产环境建议指定具体域名
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
            print(f"[SUCCESS] 准备返回响应: {response_data}")
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
        print(f"[ERROR] 异常详情: {str(e)}")
        print(f"[ERROR] 异常类型: {type(e)}")
        logger.error(f"保存设备信息异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存设备信息失败: {str(e)}")

# 添加健康检查和测试端点
@app.get("/api/test")
async def test_endpoint():
    """测试端点，验证API可达性"""
    return {
        "status": "success",
        "message": "HTTP API测试成功",
        "timestamp": datetime.now().isoformat(),
        "server_info": {
            "protocol": "http",
            "host": "127.0.0.1",  # 只监听本地
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

if __name__ == "__main__":
    print("🚀 启动AeroSense设备配网API服务...")
    print("📡 服务地址: http://127.0.0.1:8889")  # 改为HTTP
    print("📋 API文档: http://127.0.0.1:8889/docs")
    print("🔒 安全模式: 只监听本地接口")
    
    # HTTP配置 - 只监听本地（推荐）
    uvicorn.run(
        app, 
        host="127.0.0.1",  # 只监听本地，提高安全性
        port=8889,
        log_level="info",
        reload=False
        # 移除SSL配置
    )