import io
import tempfile
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


from whoami.tool.agent.tool import SenseVoiceAsr
sensevoice = SenseVoiceAsr()


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



@app.post("/transcribe/test", response_model=TranscriptionResponse)
async def transcribe_audio_test(
    request: Request, 
    audio_file: UploadFile = None,
    file_path=None,
):
    try:
        transcription_result =  await sensevoice.execute(audio_data=audio_file, file_path=file_path)
    except Exception as e:
        return TranscriptionResponse(
            text=str(e),
            status="failed"
        )
    return TranscriptionResponse(
        text=transcription_result,
        status="success"
    )



if __name__ == "__main__":
    uvicorn.run("whoami.scripts.tts.asr_scripts:app", host="0.0.0.0", port=8818, reload=True)