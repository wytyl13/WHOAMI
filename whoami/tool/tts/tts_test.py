import sys
# 添加 CosyVoice 路径
sys.path.append('/work/soft/CosyVoice')
sys.path.append('/work/soft/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2(
    '/work/soft/CosyVoice/pretrained_models/CosyVoice2-0.5B',
    load_jit=True,
    load_trt=True,
    fp16=True,
    use_flow_cache=False
)
prompt_speech_16k = load_wav('/work/ai/WHOAMI/zero_shot_0.wav', 16000)

def text_generator():
    yield '您昨天晚上的睡眠情况很好哦！'
    yield '您昨天晚上的睡眠情况很好哦！'
    yield '您昨天晚上的睡眠情况很好哦！'
    yield '您昨天晚上的睡眠情况很好哦！'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)