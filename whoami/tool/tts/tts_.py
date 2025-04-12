import sys
import os
import uuid
import shutil
import re
import torch
import torchaudio
import logging
import asyncio
from datetime import datetime
import concurrent.futures
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Generator, Tuple
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cosyvoice-api")

# 添加 CosyVoice 路径
sys.path.append('/work/soft/CosyVoice')
sys.path.append('/work/soft/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.utils.file_utils import load_wav

# 导入 CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
except ImportError as e:
    print(f"导入 CosyVoice 失败: {e}")


# 配置参数
MAX_TEXT_LENGTH = 2000  # 单次请求最大文本长度
CHUNK_SIZE = 30  # 分块大小，确保单次处理的文本不超过30字符
DEFAULT_STYLE = "希望你以后能够做的比我还好呦。"  # 默认语音风格
CACHE_DIR = "cache"  # 缓存目录
BATCH_SIZE = 5  # 批处理大小（超参数）- 同时处理的句子数量


# 创建必要的目录
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 配置线程池
thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE)


# 全局模型对象
cosyvoice = None

# 在应用启动时加载模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    await load_model()
    yield
    logger.info("服务关闭，资源已清理")
    
app = FastAPI(
    title="CosyVoice2 TTS API",
    description="语音合成API服务，支持长文本",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 挂载静态文件目录，使输出文件可以直接通过URL访问
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

async def load_model():
    """异步加载模型"""
    global cosyvoice
    if cosyvoice is not None:
        return
        
    try:
        # 在加载模型前释放缓存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 加载模型
        loop = asyncio.get_event_loop()
        cosyvoice = await loop.run_in_executor(
            None,
            lambda: CosyVoice2(
                '/work/soft/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                load_jit=True,
                load_trt=True,
                fp16=True,
                use_flow_cache=False
            )
        )
        
        logger.info("CosyVoice2模型已成功加载！")
        
        # 加载后再次清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"加载CosyVoice2模型时出错: {e}")
        raise


# def load_wav(file_path: str, sample_rate: int = 16000) -> torch.Tensor:
#     """加载WAV文件为单声道，并进行必要的前处理"""
#     try:
#         waveform, sr = torchaudio.load(file_path)
        
#         # 确保是单声道
#         if waveform.shape[0] > 1:  # 如果是多声道（例如立体声）
#             waveform = torch.mean(waveform, dim=0, keepdim=True)  # 将多声道平均为单声道
            
#         # 调整采样率
#         if sr != sample_rate:
#             waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
#         # 规范化音频长度（可选）
#         max_length = 10 * sample_rate  # 最多10秒
#         if waveform.shape[1] > max_length:
#             waveform = waveform[:, :max_length]

#         #######################################
#         # 新增：保存预处理后的音频（调试用）
#         #######################################
#         # 生成带时间戳的文件名
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         base_name = os.path.basename(file_path)
#         save_path = f"{os.path.splitext(base_name)[0]}_processed_{timestamp}.wav"
        
#         # 保存为16-bit PCM格式（兼容性最好）
#         torchaudio.save(
#             save_path,
#             waveform,
#             sample_rate,
#             encoding="PCM_S",
#             bits_per_sample=16
#         )
#         logger.debug(f"预处理音频已保存至: {save_path}")
#         #######################################
            
#         return waveform
#     except Exception as e:
#         logger.error(f"加载音频文件时出错: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"无法处理音频文件: {str(e)}")



# def load_wav(file_path: str, sample_rate: int = 16000) -> torch.Tensor:
#     """加载WAV文件为单声道，并进行必要的前处理"""
#     try:
#         waveform, sr = torchaudio.load(file_path)
        
#         # 确保是单声道
#         if waveform.shape[0] > 1:  # 如果是多声道（例如立体声）
#             waveform = torch.mean(waveform, dim=0, keepdim=True)  # 将多声道平均为单声道
            
#         # 调整采样率
#         if sr != sample_rate:
#             waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
#         # 规范化音频长度（可选）
#         max_length = 10 * sample_rate  # 最多10秒
#         if waveform.shape[1] > max_length:
#             waveform = waveform[:, :max_length]
            
#         return waveform
#     except Exception as e:
#         logger.error(f"加载音频文件时出错: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"无法处理音频文件: {str(e)}")

def split_text(text: str) -> List[str]:
    """智能分割文本为句子，确保每段不超过CHUNK_SIZE字符"""
    # 处理中文和英文的标点符号
    sentences = re.split(r'([。！？；：\.!\?;:])', text)
    result = []
    
    # 将分割的标点符号重新附加到前一个分段
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])
    
    # 如果最后一个元素不是标点符号，添加它
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
        
    # 过滤掉空字符串，并处理过长的句子
    processed = []
    for sentence in result:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 处理过长的句子，先按逗号等次要标点符号分割
        if len(sentence) > CHUNK_SIZE:
            subparts = re.split(r'([，、,])', sentence)
            subresult = []
            
            for j in range(0, len(subparts) - 1, 2):
                if j + 1 < len(subparts):
                    subresult.append(subparts[j] + subparts[j + 1])
                else:
                    subresult.append(subparts[j])
                    
            if len(subparts) % 2 == 1:
                subresult.append(subparts[-1])
                
            # 进一步处理仍然过长的片段
            for subsentence in subresult:
                if len(subsentence) > CHUNK_SIZE:
                    # 按字符硬切分
                    for k in range(0, len(subsentence), CHUNK_SIZE):
                        chunk = subsentence[k:k+CHUNK_SIZE]
                        if chunk:
                            processed.append(chunk)
                else:
                    processed.append(subsentence)
        else:
            processed.append(sentence)
    
    return processed


async def process_sentence(
    sentence: str,
    sentence_idx: int,
    prompt_speech: torch.Tensor,
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> Tuple[int, List[Dict[str, Any]]]:
    """处理单个句子并返回结果（带索引）"""
    try:
        logger.info(f"处理句子 (索引: {sentence_idx}): '{sentence}'")
        
        # 使用适当的推理方法
        if instruct:
            results = list(cosyvoice.inference_instruct2(
                sentence, instruct, prompt_speech, stream=stream, speed=speed
            ))
        else:
            results = list(cosyvoice.inference_zero_shot(
                sentence, style, prompt_speech, stream=stream, speed=speed
            ))
        
        return sentence_idx, results
    except Exception as e:
        logger.error(f"处理句子 '{sentence}' (索引: {sentence_idx}) 时出错: {str(e)}")
        return sentence_idx, []  # 返回空结果列表表示处理失败


def process_sentence_sync(
    sentence: str,
    sentence_idx: int,
    prompt_speech: torch.Tensor,
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    同步处理单个句子（用于在线程池中执行）
    """
    try:
        logger.info(f"处理句子 (索引: {sentence_idx}): '{sentence}'")
        
        # 使用适当的推理方法
        if instruct:
            logger.info("instruct --------------------------------------------------")
            results = list(cosyvoice.inference_instruct2(
                sentence, instruct, prompt_speech, stream=stream, speed=speed
            ))
        else:
            logger.info("zero shot --------------------------------------------------")
            results = list(cosyvoice.inference_zero_shot(
                sentence, style, prompt_speech, stream=stream, speed=speed
            ))
        
        return sentence_idx, results
    except Exception as e:
        logger.error(f"处理句子 '{sentence}' (索引: {sentence_idx}) 时出错: {str(e)}")
        return sentence_idx, []  # 返回空结果列表表示处理失败


async def process_sentence_batch_multi_thread(
    batch_sentences: List[str],
    batch_indices: List[int],
    prompt_speech: torch.Tensor,
    style: str = # The above code is a comment in Python. Comments are used to provide explanations or
    # notes within the code for better understanding. In this case, the comment appears
    # to be indicating a default style setting.
    DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Tuple[int, Dict[str, Any]]]:
    """使用多线程并行处理一批文本句子并返回结果"""
    batch_results = []
    loop = asyncio.get_event_loop()
    
    # 准备多线程任务
    futures = []
    for i, (sentence_idx, sentence) in enumerate(zip(batch_indices, batch_sentences)):
        future = loop.run_in_executor(
            thread_pool_executor,
            process_sentence_sync,
            sentence,
            sentence_idx,
            prompt_speech,
            style,
            instruct,
            stream,
            speed
        )
        futures.append(future)
    
    # 等待所有任务完成
    results = await asyncio.gather(*futures)
    
    # 处理结果
    for sentence_idx, sentence_results in results:
        for result in sentence_results:
            batch_results.append((sentence_idx, result))
    
    return batch_results


async def process_sentence_batch(
    batch_sentences: List[str],
    batch_indices: List[int],
    prompt_speech: torch.Tensor,
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Tuple[int, Dict[str, Any]]]:
    """并行处理一批文本句子并返回结果"""
    # 创建任务列表
    tasks = []
    for i, (sentence_idx, sentence) in enumerate(zip(batch_indices, batch_sentences)):
        # 为每个句子创建异步任务
        task = process_sentence(
            sentence, sentence_idx, prompt_speech, style, instruct, stream, speed
        )
        tasks.append(task)
    
    # 并行执行所有任务
    batch_results = []
    results = await asyncio.gather(*tasks)
    
    # 处理结果
    for sentence_idx, sentence_results in results:
        # 将每个句子的结果与其索引一起添加到批次结果中
        for result in sentence_results:
            batch_results.append((sentence_idx, result))
    
    return batch_results


async def process_long_text_batch(
    text: str, 
    prompt_speech: torch.Tensor, 
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Dict[str, Any]]:
    """处理长文本并返回结果列表，使用批处理并行处理"""
    if not text or prompt_speech is None:
        raise ValueError("文本和提示音频不能为空")
        
    # 检查模型是否已加载
    if cosyvoice is None:
        await load_model()
        if cosyvoice is None:
            raise HTTPException(status_code=503, detail="TTS模型未加载")
    
    # 分割长文本为小片段
    sentences = split_text(text)
    logger.info(f"文本已分割为 {len(sentences)} 个片段，使用批处理大小: {BATCH_SIZE}")
    
    # 创建批次
    batches = []
    batch_indices = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        indices = list(range(i, min(i+BATCH_SIZE, len(sentences))))
        batches.append(batch)
        batch_indices.append(indices)
    
    logger.info(f"创建了 {len(batches)} 个批次进行处理")
    
    # 存储所有结果（带索引，以确保正确顺序）
    all_indexed_results = []
    
    # 逐批处理
    for batch_num, (batch, indices) in enumerate(zip(batches, batch_indices)):
        try:
            logger.info(f"开始处理第 {batch_num+1}/{len(batches)} 批句子")
            
            # 处理当前批次（多线程并行）
            batch_results = await process_sentence_batch_multi_thread(
                batch, indices, prompt_speech, style, instruct, stream, speed
            )
            
            # 添加到结果列表
            all_indexed_results.extend(batch_results)
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"处理第 {batch_num+1} 批次时出错: {str(e)}")
            # 继续处理其他批次
    
    # 按原始索引排序结果，确保正确顺序
    all_indexed_results.sort(key=lambda x: x[0])
    
    # 返回排序后的结果列表（仅保留结果部分，丢弃索引）
    return [result for _, result in all_indexed_results]



def optimize_speech_processing(all_speeches: List[torch.Tensor], silence_duration: float = 0.05, trim_threshold: float = 0.01) -> torch.Tensor:
    """
    优化语音片段合并，包括:
    1. 修剪每个片段末尾的静音
    2. 标准化音量
    3. 精确控制片段间间隔
    4. 应用平滑过渡
    
    参数:
    - all_speeches: 所有语音片段列表
    - silence_duration: 片段之间的静音持续时间(秒)，默认0.05秒
    - trim_threshold: 用于检测静音的振幅阈值，默认0.01
    """
    if not all_speeches:
        return None
    
    # 获取采样率（CosyVoice的默认采样率通常是16000Hz）
    sample_rate = 16000
    silence_samples = int(silence_duration * sample_rate)
    
    # 检查和过滤无效张量
    valid_speeches = []
    for speech in all_speeches:
        if isinstance(speech, torch.Tensor) and speech.numel() > 0:
            valid_speeches.append(speech)
    
    if not valid_speeches:
        return None
    
    # 修剪和标准化每个片段
    processed_speeches = []
    for speech in valid_speeches:
        # 标准化音量
        rms = torch.sqrt(torch.mean(speech ** 2))
        if rms > 0:
            # 应用标准化，目标RMS为0.05
            target_rms = 0.05
            speech = speech * (target_rms / rms)
        
        # 修剪末尾静音
        # 从末尾向前扫描，找到第一个超过阈值的样本
        abs_speech = torch.abs(speech[0])  # 假设是单声道
        non_silent_pos = torch.where(abs_speech > trim_threshold)[0]
        
        if len(non_silent_pos) > 0:
            # 找到最后一个非静音样本的位置
            last_sound_pos = non_silent_pos[-1].item()
            # 保留一点点尾音（例如50ms）以确保自然度
            tail_samples = int(0.05 * sample_rate)
            end_pos = min(last_sound_pos + tail_samples, speech.shape[1])
            # 修剪片段
            speech = speech[:, :end_pos]
        
        processed_speeches.append(speech)
    
    # 如果只有一个语音片段，直接返回
    if len(processed_speeches) == 1:
        return processed_speeches[0]
    
    # 创建合并片段，控制间隔时间
    combined_speeches = []
    for i, speech in enumerate(processed_speeches):
        combined_speeches.append(speech)
        
        # 在除最后一个片段外的每个片段后添加指定长度的静音
        if i < len(processed_speeches) - 1:
            silence = torch.zeros((speech.shape[0], silence_samples), dtype=speech.dtype, device=speech.device)
            combined_speeches.append(silence)
    
    # 合并所有片段
    all_speech = torch.cat(combined_speeches, dim=1)
    
    # 应用跨片段平滑（可选，根据需要）
    # 这部分可能需要根据实际效果调整
    
    return all_speech


def optimize_speech_processing_bake(all_speeches: List[torch.Tensor], trim_threshold: float = 0.01, crossfade_duration: float = 0.03) -> torch.Tensor:
    """
    优化语音片段无缝合并，包括:
    1. 彻底修剪每个片段开头和结尾的静音
    2. 标准化音量
    3. 使用交叉淡入淡出技术实现平滑过渡
    4. 零间隔连接
    
    参数:
    - all_speeches: 所有语音片段列表
    - trim_threshold: 用于检测静音的振幅阈值，默认0.01
    - crossfade_duration: 交叉淡变持续时间(秒)，默认0.03秒
    """
    if not all_speeches:
        return None
    
    # 获取采样率（CosyVoice的默认采样率通常是16000Hz）
    sample_rate = 16000
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # 检查和过滤无效张量
    valid_speeches = []
    for speech in all_speeches:
        if isinstance(speech, torch.Tensor) and speech.numel() > 0:
            valid_speeches.append(speech)
    
    if not valid_speeches:
        return None
    
    # 修剪和标准化每个片段
    processed_speeches = []
    for speech in valid_speeches:
        # 标准化音量
        rms = torch.sqrt(torch.mean(speech ** 2))
        if rms > 0:
            # 应用标准化，目标RMS为0.05
            target_rms = 0.05
            speech = speech * (target_rms / rms)
        
        # 修剪前后静音
        abs_speech = torch.abs(speech[0])  # 假设是单声道
        
        # 找到所有非静音样本的位置
        non_silent_pos = torch.where(abs_speech > trim_threshold)[0]
        
        if len(non_silent_pos) > 0:
            # 找到第一个和最后一个非静音样本的位置
            first_sound_pos = non_silent_pos[0].item()
            last_sound_pos = non_silent_pos[-1].item()
            
            # 在开头保留少量前导（10ms），以避免切得太死
            start_pos = max(0, first_sound_pos - int(0.01 * sample_rate))
            
            # 在结尾保留少量尾音（20ms），以保持自然感
            end_pos = min(last_sound_pos + int(0.02 * sample_rate), speech.shape[1])
            
            # 修剪片段
            speech = speech[:, start_pos:end_pos]
        
        processed_speeches.append(speech)
    
    # 如果只有一个语音片段，直接返回
    if len(processed_speeches) == 1:
        return processed_speeches[0]
    
    # 使用交叉淡变实现无缝连接
    combined_speech = processed_speeches[0]
    
    for i in range(1, len(processed_speeches)):
        next_speech = processed_speeches[i]
        
        # 确保交叉淡变区域不超过当前片段的长度
        actual_crossfade = min(crossfade_samples, combined_speech.shape[1], next_speech.shape[1])
        
        if actual_crossfade > 0:
            # 创建淡出和淡入曲线
            fade_out = torch.linspace(1, 0, actual_crossfade)
            fade_in = torch.linspace(0, 1, actual_crossfade)
            
            # 应用淡出到当前合并片段的末尾
            combined_end = combined_speech[:, -actual_crossfade:]
            faded_end = combined_end * fade_out
            
            # 应用淡入到下一个片段的开头
            next_start = next_speech[:, :actual_crossfade]
            faded_start = next_start * fade_in
            
            # 混合交叉区域
            crossfade_region = faded_end + faded_start
            
            # 创建新的合并片段，去掉当前片段的交叉区域，添加混合区域和下一个片段的余下部分
            new_combined = torch.cat([
                combined_speech[:, :-actual_crossfade],
                crossfade_region,
                next_speech[:, actual_crossfade:]
            ], dim=1)
            
            combined_speech = new_combined
        else:
            # 如果交叉淡变不可能，直接连接
            combined_speech = torch.cat([combined_speech, next_speech], dim=1)
    
    return combined_speech


async def process_long_text(
    text: str, 
    prompt_speech: torch.Tensor, 
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Dict[str, Any]]:
    """处理长文本并返回结果列表"""
    if not text or prompt_speech is None:
        raise ValueError("文本和提示音频不能为空")
        
    # 检查模型是否已加载
    if cosyvoice is None:
        await load_model()
        if cosyvoice is None:
            raise HTTPException(status_code=503, detail="TTS模型未加载")
    
    # 分割长文本为小片段
    sentences = split_text(text)
    logger.info(f"文本已分割为 {len(sentences)} 个片段")
    logger.info(f"sentences: {sentences} ")
    
    # 存储所有结果
    all_results = []
    
    # 逐段处理
    for i, sentence in enumerate(sentences):
        try:
            # 清理显存
            if i > 0 and i % 3 == 0:  # 每处理3个片段清理一次显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # 使用适当的推理方法
            if instruct:
                results = list(cosyvoice.inference_instruct2(
                    # The above code is not valid Python code. It seems to contain some random text
                    # ("sentence") and comment symbols ("
                    sentence, instruct, prompt_speech, stream=stream, speed=speed
                ))
            else:
                results = list(cosyvoice.inference_zero_shot(
                    sentence, style, prompt_speech, stream=stream, speed=speed
                ))
            
            # 添加到结果列表
            if results and len(results) > 0:
                all_results.extend(results)
                
            # 在控制台输出进度
            logger.info(f"已处理 {i+1}/{len(sentences)} 个片段: '{sentence}'")
            
        except Exception as e:
            logger.error(f"处理第 {i+1} 个文本片段 '{sentence}' 时出错: {str(e)}")
            # 继续处理其他片段
            continue
    
    return all_results


# prompt_speech_16k = load_wav('/work/soft/CosyVoice/asset/zero_shot_prompt.wav', 16000)
prompt_speech_16k = load_wav('/work/ai/WHOAMI/whoami/tool/tts/ZH_2_prompt.wav', 16000)



# 创建请求模型
class TTSRequest(BaseModel):
    text: str
    style: str = DEFAULT_STYLE
    instruct: Optional[str] = None
    wait_complete: bool = False
    speed: float = 1.0
    use_batch: bool = False


@app.post("/tts")
async def tts(
    request: Request,
    background_tasks: BackgroundTasks,
    tts_request: TTSRequest,
):
    # 在这里处理请求
    text = tts_request.text
    style = tts_request.style
    instruct = tts_request.instruct
    wait_complete = tts_request.wait_complete
    speed = tts_request.speed
    use_batch = tts_request.use_batch
    """长文本语音合成API，可选择异步处理和批处理
    非流式生成，后续添加流式生成，使用generator
    """
    # 验证速度参数
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="速度参数必须在0.5到2.0之间")
    prompt_audio = None
    prompt_path = None
    final_prompt_speech_16k = None
    try:
        # 加载提示音频
        if prompt_audio is not None:
            # 保存上传的音频文件
            prompt_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
            with open(prompt_path, "wb") as buffer:
                shutil.copyfileobj(prompt_audio.file, buffer)
            final_prompt_speech_16k = load_wav(prompt_path, 16000)
        else:
            # 使用默认的提示音频
            final_prompt_speech_16k = prompt_speech_16k
        
        # 生成任务ID和输出文件名
        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 构建结果URL
        base_url = str(request.base_url).rstrip('/')
        audio_url = f"{base_url}/audio/{output_filename}"
        
        # 定义异步处理函数
        async def process_text_task():
            try:
                # 根据use_batch选择处理方法
                if use_batch:
                    logger.info(f"使用批处理模式处理文本，批处理大小: {BATCH_SIZE}")
                    results = await process_long_text_batch(
                        text, 
                        final_prompt_speech_16k, 
                        style, 
                        instruct, 
                        stream=False, 
                        speed=speed
                    )
                else:
                    logger.info("使用常规模式处理文本（不使用批处理）")
                    results = await process_long_text(
                        text, 
                        final_prompt_speech_16k, 
                        style, 
                        instruct, 
                        stream=False, 
                        speed=speed
                    )
                
                # 合并所有音频片段，使用优化的音频处理
                if results and len(results) > 0:
                    all_speeches = [result['tts_speech'] for result in results if 'tts_speech' in result]
                    if all_speeches:
                        # 使用较短的间隔时间，例如0.1秒
                        all_speech = optimize_speech_processing(all_speeches, silence_duration=0.001)
                        if all_speech is not None:
                            torchaudio.save(output_path, all_speech, cosyvoice.sample_rate)
                            
                            # 保存任务状态
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("completed")
                        else:
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("error: 音频处理失败")
                    else:
                        with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                            f.write("error: 未生成有效的语音片段")
                else:
                    # 没有结果时保存错误状态
                    with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                        f.write("error: 未生成有效的TTS结果")
                        
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"异步处理任务 {task_id} 时出错: {str(e)}")
                # 保存错误状态
                with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                    f.write(f"error: {str(e)}")
            finally:
                # 清理临时文件
                if prompt_path is not None and os.path.exists(prompt_path):
                    os.remove(prompt_path)
        
        if wait_complete:
            # 同步处理
            await process_text_task()
            return {
                "success": True,
                "task_id": task_id,
                "audio_url": audio_url,
                "status": "completed",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "use_batch": use_batch,
                "batch_size": BATCH_SIZE if use_batch else None,
                "message": "长文本语音合成成功"
            }
        else:
            # 异步处理
            background_tasks.add_task(process_text_task)
            
            # 初始化任务状态
            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                f.write("processing")
                
            return {
                "success": True,
                "task_id": task_id,
                "status": "processing",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "use_batch": use_batch,
                "batch_size": BATCH_SIZE if use_batch else None,
                "check_status_url": f"{base_url}/status/{task_id}",
                "message": "长文本语音合成任务已开始处理"
            }
    
    except Exception as e:
        logger.error(f"处理长文本请求时出错: {str(e)}")
        # 清理临时文件
        if prompt_path is not None and os.path.exists(prompt_path):
            os.remove(prompt_path)
        raise HTTPException(status_code=500, detail=f"TTS处理错误: {str(e)}")



@app.post("/tts/bake")
async def tts_bake(
    request: Request,
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_audio: Optional[UploadFile] = File(None),
    style: str = Form(DEFAULT_STYLE),
    instruct: Optional[str] = Form(None),
    wait_complete: bool = Form(False),
    speed: float = Form(1.0)
):
    """长文本语音合成API，可选择异步处理
    非流式生成，后续添加流式生成，使用generator
    """
    # 验证速度参数
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="速度参数必须在0.5到2.0之间")
    
    prompt_path = None
    try:
        # 加载提示音频
        if prompt_audio is not None:
            # 保存上传的音频文件
            prompt_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
            with open(prompt_path, "wb") as buffer:
                shutil.copyfileobj(prompt_audio.file, buffer)
            prompt_speech_16k = load_wav(prompt_path, 16000)
        else:
            # 使用默认的提示音频
            prompt_speech_16k = load_wav('/work/soft/CosyVoice/asset/zero_shot_prompt.wav', 16000)
        
        # 生成任务ID和输出文件名
        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 构建结果URL
        base_url = str(request.base_url).rstrip('/')
        audio_url = f"{base_url}/audio/{output_filename}"
        # 定义异步处理函数
        async def process_text_task():
            try:
                # 处理文本
                results = await process_long_text(text, prompt_speech_16k, style, instruct, stream=False, speed=speed)
                
                # 合并所有音频片段，使用优化的音频处理
                if results and len(results) > 0:
                    all_speeches = [result['tts_speech'] for result in results if 'tts_speech' in result]
                    if all_speeches:
                        # 使用较短的间隔时间，例如0.1秒
                        all_speech = optimize_speech_processing(all_speeches, silence_duration=0.05)
                        if all_speech is not None:
                            torchaudio.save(output_path, all_speech, cosyvoice.sample_rate)
                            
                            # 保存任务状态
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("completed")
                        else:
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("error: 音频处理失败")
                    else:
                        with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                            f.write("error: 未生成有效的语音片段")
                else:
                    # 没有结果时保存错误状态
                    with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                        f.write("error: 未生成有效的TTS结果")
                        
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"异步处理任务 {task_id} 时出错: {str(e)}")
                # 保存错误状态
                with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                    f.write(f"error: {str(e)}")
            finally:
                # 清理临时文件
                if prompt_path is not None and os.path.exists(prompt_path):
                    os.remove(prompt_path)
        
        if wait_complete:
            # 同步处理
            await process_text_task()
            return {
                "success": True,
                "task_id": task_id,
                "audio_url": audio_url,
                "status": "completed",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "message": "长文本语音合成成功"
            }
        else:
            # 异步处理
            background_tasks.add_task(process_text_task)
            
            # 初始化任务状态
            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                f.write("processing")
                
            return {
                "success": True,
                "task_id": task_id,
                "status": "processing",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "check_status_url": f"{base_url}/status/{task_id}",
                "message": "长文本语音合成任务已开始处理"
            }
    
    except Exception as e:
        logger.error(f"处理长文本请求时出错: {str(e)}")
        # 清理临时文件
        if os.path.exists(prompt_path):
            os.remove(prompt_path)
        raise HTTPException(status_code=500, detail=f"TTS处理错误: {str(e)}")



@app.get("/status/{task_id}")
async def check_task_status(request: Request, task_id: str):
    """检查长文本处理任务的状态"""
    status_path = os.path.join(CACHE_DIR, f"{task_id}.status")
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    # 读取任务状态
    with open(status_path, "r") as f:
        status = f.read().strip()
    
    base_url = str(request.base_url).rstrip('/')
    
    if status == "completed" and os.path.exists(output_path):
        return {
            "task_id": task_id,
            "status": status,
            "audio_url": f"{base_url}/audio/{task_id}.wav",
            "completed": True
        }
    elif status.startswith("error"):
        return {
            "task_id": task_id,
            "status": "error",
            "error_message": status[7:] if len(status) > 7 else "未知错误",
            "completed": True
        }
    else:
        return {
            "task_id": task_id,
            "status": status,
            "completed": False
        }



@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用CosyVoice2 TTS API服务",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/tts", "method": "POST", "description": "标准语音合成"},
            {"path": "/tts_long", "method": "POST", "description": "长文本语音合成，支持异步处理"},
            {"path": "/status/{task_id}", "method": "GET", "description": "检查长文本处理任务的状态"}
        ],
        "status": "ready" if cosyvoice is not None else "initializing"
    }




if __name__ == "__main__":
    import time  # 添加缺少的导入
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)