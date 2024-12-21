
import asyncio


# from ai_search.tool.google_search import GoogleSearch
# from ai_search.utils.yaml_model import YamlModel
# from pathlib import Path
# from ai_search.llm_api.ollama_llm import OllamLLM
# from ai_search.configs.llm_config import LLMConfig
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.tool.detect.sx_video_stream_detector import SxVideoStreamDetector
# from ai_search.tool.detect.sx_detector_warning import SxDetectorWarning
#1 async def test():
#     llm = OllamLLM(LLMConfig.from_file(Path("/home/weiyutao/.metagpt/config2.yaml")))
#     content = await llm.whoami("我是谁")
#     return content

# async def main():
#     content = await test()  # 使用 await 调用异步函数
#     print(content)

if __name__ == '__main__':
    model_path = '/work/ai/object_detect_warning/models/yolov10m_20000_fire_epoch_250.pt'
    # yolo = UltraliticsDetector(model_path=model_path)
    video_stream_detector = SxVideoStreamDetector(device_sn='BD3202818', topic_name='/fire/smoke/warning', url_str_flag='old')
    print(video_stream_detector.process())
    # print(yolo.name)
    # print(yolo.model.info())
    # image_path = '/work/ai/object_detect_warning/data/12月17日(12).jpg'
    # result = yolo.predict(image_path)
    # print(result)
    
    
    
    
    
    # google_search = GoogleSearch(snippet_flag=1)
    # param = {
    #     "query": "我是谁"
    # }
    # status, result = google_search(**param)
    # print(result)
    
    # print(YamlModel.read(Path('/home/weiyutao/.metagpt/config2.yaml')))
    # asyncio.run(main())  # 使用 asyncio.run 来运行主程序