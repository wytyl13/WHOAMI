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





class ChatInfo(BaseModel):
    question: Optional[str] = None


app = FastAPI(title="治愈api", version="1.0.0")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

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
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8818,
        log_level="info",
        reload=False,
    )