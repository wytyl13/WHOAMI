import asyncio
import json
import websockets
import base64
import wave
import io
import os

async def test_text_to_speech():
    uri = "ws://localhost:3000"
    async with websockets.connect(uri) as websocket:
        # 发送文本请求
        await websocket.send(json.dumps({
            "type": "text",
            "text": "当地时间4月18日，美国哥伦比亚广播公司报道称，多名消息人士透露，由于特朗普政府对中国商品加征的畸高关税将导致供应链危机，特朗普政府内部已开始讨论组建一个工作组，以便在未能与中国政府谈判取得突破的情况下紧急处理这些问题。消息人士称，目前有关成立该小组的具体情况还未敲定任何细节，但工作组可能包括副总统万斯、财政部长贝森特、商务部长卢特尼克、白宫国家经济委员会主任凯文·哈西特、白宫经济顾问委员会主席斯蒂芬·米兰和美国贸易代表贾米森·格里尔。针对美方对华加征畸高关税，我外交部已多次回应。4月17日，外交部发言人再次强调，美方对华轮番加征畸高关税已经沦为数字游戏，在经济上已无实际意义，只会更加暴露出美方将关税工具化、武器化，搞霸凌胁迫的伎俩。关税战、贸易战没有赢家，中方不愿打，但也不怕打。如果美方继续玩弄关税数字游戏，中方将不予理会。倘若美方执意继续实质性侵害中方权益，中方将坚决反制，奉陪到底。"
        }))
        
        # 接收响应
        audio_chunks = []
        chunk_texts = []
        chunk_count = 0
        index = 0
        while True:
            response = await websocket.recv()
            
            # 检查是否是JSON消息
            try:
                data = json.loads(response)
                print(f"收到消息: {data}")
                
                if data.get("status") == "complete":
                    print("处理完成")
                    break
                    
                if data.get("status") == "chunk":
                    chunk_count += 1
                    if "chunk_text" in data:
                        chunk_texts.append(data["chunk_text"])
                    
                    # 下一条消息应该是二进制音频数据
                    audio_data = await websocket.recv()
                    audio_chunks.append(audio_data)
                    print(f"收到音频块 {chunk_count}, 大小: {len(audio_data)} 字节")
                    with open(f"test_audio_{index+1}.wav", "wb") as f:
                        f.write(audio_data)
                    print(f"保存音频块 {index+1} 到 test_audio_{index+1}.wav")
                    index += 1
            except json.JSONDecodeError:
                # 这可能是二进制音频数据
                print(f"收到二进制数据，大小: {len(response)} 字节")
        

# 运行测试
asyncio.run(test_text_to_speech())