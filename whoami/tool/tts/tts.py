import io
import tempfile
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from modelscope.hub.snapshot_download import snapshot_download

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="语音识别 API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionResponse(BaseModel):
    text: str
    status: str

def load_sensevoice_model():
    """
    加载SenseVoice模型，处理可能的错误
    """
    try:
        from funasr import AutoModel
        
        # 有两种方式：
        # 1. 如果模型已经下载到本地，直接使用本地路径
        model_local_path = '/root/.cache/modelscope/hub/models/iic/SenseVoiceSmall'
        if os.path.exists(model_local_path):
            logger.info(f"使用本地模型路径: {model_local_path}")
            model_dir = model_local_path
        else:
            # 2. 如果本地没有，从modelscope下载模型
            logger.info("本地未找到模型，从modelscope下载")
            model_dir = snapshot_download('iic/SenseVoiceSmall')
            logger.info(f"模型已下载到: {model_dir}")
        
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            disable_update=True,  # 禁用更新检查
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )
        
        logger.info("SenseVoice模型加载成功")
        return model
    except Exception as e:
        logger.error(f"加载SenseVoice模型失败: {str(e)}")
        raise e

# 全局模型变量
model = None

@app.on_event("startup")
async def startup_event():
    """
    服务启动时加载模型
    """
    global model
    try:
        model = load_sensevoice_model()
    except Exception as e:
        logger.error(f"启动时加载模型失败: {str(e)}")
        # 不抛出异常，让服务继续启动，后续请求会再次尝试加载

async def process_audio(audio_bytes: io.BytesIO):
    """
    处理音频数据的核心功能
    """
    global model
    
    # 如果模型未加载，尝试再次加载
    if model is None:
        model = load_sensevoice_model()
    
    try:
        # 保存音频数据到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes.getvalue())
        
        try:
            # 使用SenseVoice处理音频
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            
            res = model.generate(
                input=temp_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            transcription = rich_transcription_postprocess(res[0]["text"])
            logger.info(f"转录成功: {transcription[:50]}...")
            
            return transcription
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"转录过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音处理错误: {str(e)}")

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(request: Request, audio_file: UploadFile = None):
    """
    统一接口 - 可以处理上传的文件或原始音频数据
    """
    try:
        # 检查内容类型
        content_type = request.headers.get("content-type", "")
        
        # 处理不同情况的音频数据
        if audio_file is not None:
            # 处理上传的文件
            logger.info(f"接收到音频文件: {audio_file.filename}")
            content = await audio_file.read()
            audio_bytes = io.BytesIO(content)
        elif content_type.startswith("audio/"):
            # 处理原始音频数据
            content = await request.body()
            logger.info(f"接收到原始音频数据: {len(content)} 字节")
            audio_bytes = io.BytesIO(content)
        else:
            # 检查是否有请求体但格式不是 multipart/form-data 或 audio/*
            content = await request.body()
            if content:
                logger.info(f"接收到未知格式的音频数据: {len(content)} 字节")
                audio_bytes = io.BytesIO(content)
            else:
                raise HTTPException(status_code=400, detail="没有提供音频数据")
        
        # 使用共用函数处理音频
        transcription = await process_audio(audio_bytes)
        
        return TranscriptionResponse(
            text=transcription,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("tts:app", host="0.0.0.0", port=8818, reload=True)