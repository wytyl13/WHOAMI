# 继续使用你原有的 YamlModel 类
from whoami.utils.yaml_model import YamlModel

class PromptTemplatesConfig:
    """
    提示词模板配置，使用 YamlModel 加载数据
    """
    def __init__(self, file_path=None):
        self.data = YamlModel.read(file_path)
        
    def get_prompt(self, task_name):
        """获取任务的提示词模板"""
        task_config = self.data.get(task_name, {})
        return task_config.get('prompt', '')
        
    def get_default_result(self, task_name):
        """获取任务的默认结果"""
        task_config = self.data.get(task_name, {})
        return task_config.get('default_result', {})
        
    def get_task_names(self):
        """获取所有任务名称"""
        return list(self.data.keys())