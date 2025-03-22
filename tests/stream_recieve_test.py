import asyncio
import json
import pytest
import aiohttp


async def stream_chat_health_report_async():
    url = "http://1.71.15.121:8888/chat_health_report"
    
    payload = {
        "question": "你们的公司地址",
        "user_id": "u1ser122232",
        "conversation_id": "111",
        "messages": []
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    print("开始接收流式响应：")
    print("-" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    print(f"错误: {response.status}")
                    text = await response.text()
                    print(text)
                    return False
                
                # 处理SSE流
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:].strip()
                        if data:
                            # 每收到一块数据就立即打印出来
                            print(data, end="", flush=True)
                            # 可选：增加一个微小延迟使输出更自然
                            await asyncio.sleep(0.01)
        
        print("\n" + "-" * 50)
        print("流式响应结束")
        return True
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False


# 为pytest创建一个可测试的函数
@pytest.mark.asyncio
async def test_stream_chat_health_report():
    result = await stream_chat_health_report_async()
    assert result is True, "流式响应失败"


if __name__ == "__main__":
    asyncio.run(stream_chat_health_report_async())