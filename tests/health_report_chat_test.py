import pytest
import os
import asyncio

from whoami.tool.llm_application.health_report_chat import HealthReportChat
CONFIG_PATH = "/work/ai/WHOAMI/whoami/tool/llm_application/info_extract_prompt.yaml"

# 跳过测试如果配置文件不存在
skip_if_no_config = pytest.mark.skipif(
    not os.path.exists(CONFIG_PATH),
    reason="配置文件不存在"
)

class TestHealthReportChat:
    
    @pytest.mark.asyncio
    @skip_if_no_config
    async def test_run(self):
        """测试test_run方法"""
        print("基本测试开始执行")
        health_report = HealthReportChat()
        
        async for chunk in health_report._run(query="详细解释畅丽娟最近3天的睡眠情况，并说明哪天晚上不在床", message_history=[]):
            print(chunk, end="", flush=True)
        print()  # 打印最后的换行


if __name__ == "__main__":
    # 通过pytest模块直接运行测试
    pytest.main(["-v", __file__])
