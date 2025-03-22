import pytest
import os
import asyncio

from whoami.tool.llm_application.information_extract_json import InformationExtractJson
CONFIG_PATH = "/work/ai/WHOAMI/whoami/tool/llm_application/info_extract_prompt.yaml"

# 跳过测试如果配置文件不存在
skip_if_no_config = pytest.mark.skipif(
    not os.path.exists(CONFIG_PATH),
    reason="配置文件不存在"
)

class TestInformationExtractJson:
    
    @skip_if_no_config
    def test_initialization(self):
        """测试InformationExtractJson初始化是否正常"""
        extractor = InformationExtractJson(config_path=CONFIG_PATH)
        assert extractor is not None, "实例化失败"
        assert hasattr(extractor, 'default_results'), "缺少default_results属性"
        assert isinstance(extractor.default_results, dict), "default_results不是字典类型"

    @skip_if_no_config
    def test_default_results_content(self):
        """测试default_results内容是否符合预期"""
        extractor = InformationExtractJson(config_path=CONFIG_PATH)
        
        # 打印default_results以便观察
        print("Default results:", extractor.default_results)
        
        # 检查default_results是否非空
        assert extractor.default_results, "default_results为空"
        
        # 检查是否包含预期的任务
        # 你需要根据实际配置文件内容调整这里的任务名称
        expected_tasks = ['name_id']  # 根据你的配置文件调整
        for task in expected_tasks:
            assert task in extractor.default_results, f"缺少任务: {task}"
    
    @skip_if_no_config
    def test_prompt_templates(self):
        """测试prompt_templates内容是否符合预期"""
        extractor = InformationExtractJson(config_path=CONFIG_PATH)
        
        # 检查prompt_templates是否存在并非空
        assert hasattr(extractor, 'prompt_templates'), "缺少prompt_templates属性"
        assert extractor.prompt_templates, "prompt_templates为空"
        
        # 检查每个提示词模板是否为字符串且非空
        for task, prompt in extractor.prompt_templates.items():
            assert isinstance(prompt, str), f"任务 {task} 的提示词不是字符串"
            assert prompt, f"任务 {task} 的提示词为空"
        print(extractor.prompt_templates)
        print(extractor.default_results)

    @skip_if_no_config
    def test_parse_json_function(self):
        """测试parse_json_response方法"""
        extractor = InformationExtractJson(config_path=CONFIG_PATH)
        
        # 测试有效的JSON字符串
        valid_json = '{"key": "value", "number": 123}'
        result = extractor.parse_json_response(valid_json)
        assert result == {"key": "value", "number": 123}, "解析有效JSON失败"
        
        # 测试带有额外文本的JSON
        json_with_text = 'Some text before {"key": "value"} and after'
        result = extractor.parse_json_response(json_with_text)
        assert result == {"key": "value"}, "从文本中提取JSON失败"
        
        # 测试无效的JSON
        invalid_json = 'Not a JSON'
        default = {"default": True}
        result = extractor.parse_json_response(invalid_json, default_result=default)
        assert result == default, "处理无效JSON失败"
        
        # 测试空输入
        empty_input = ''
        result = extractor.parse_json_response(empty_input)
        assert result == {}, "处理空输入失败"


    @pytest.mark.asyncio
    @skip_if_no_config  
    async def test_extract_info(self):
        """测试extract_name_id方法"""
        print("基本测试开始执行")
        print("test_extract_info -------------------------------------------------  ")
        extractor = InformationExtractJson(config_path=CONFIG_PATH)
        assert extractor is not None, "实例化失败"
        # Add basic print statements to verify execution flow
        print("Before calling extract_name_id")
        try:
            result = await extractor.analyze_intent_health_report(query="查询下device_sn123最近两天的心率")
            print("After calling extract_name_id")
            print("Result:", result["is_related"])
            assert isinstance(result, dict), "Result should be a dictionary"
            print("test_extract_info -------------------------------------------------  ")
        except Exception as e:
            print(f"Error occurred: {e}")
            raise  # Re-raise to make the test fail properly
        


if __name__ == "__main__":
    # 通过pytest模块直接运行测试
    pytest.main(["-v", __file__])



