from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional
import sys
import os
from datetime import datetime

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
    server_ip: Optional[str] = "1.71.15.121"
    server_port: Optional[str] = "8888"
    working_distance: Optional[float] = 1.5
    scene: Optional[str] = "睡眠监测"
    username: Optional[str] = ""
    user_id: Optional[int] = None

app = FastAPI()

# 配置CORS，允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API服务运行正常"}

@app.get("/")
async def root():
    return {"message": "AeroSense设备配网API服务"}


@app.post("/api/save_device")
async def save_device(device_info: DeviceInfo):
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        print(device_info)
        # 检查现有设备
        existing_devices = sql_provider.get_record_by_condition(
            condition={"device_code": device_info.device_code}
        )
        
        # 软删除现有设备
        if existing_devices:
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
            "username": device_info.username,
            "user_id": 1,
            "status": "active",
            "creator": device_info.username,
            "updater": device_info.username,
            "deleted": False,
            "tenant_id": 0,
            "create_time": datetime.now(),
            "update_time": datetime.now()
        }
        
        # 执行数据库插入
        result = sql_provider.add_record(device_data)
        
        if result:
            return {"success": True, "message": "设备信息已成功保存", "device_code": device_info.device_code}
        else:
            raise HTTPException(status_code=500, detail="数据库添加记录失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存设备信息失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889)