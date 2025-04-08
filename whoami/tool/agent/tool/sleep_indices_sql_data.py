
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
from typing import (
    List,
    Optional,
    Dict,
    Any
)

from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.tool.agent.base_tool import tool

@tool
class SleepIndicesSqlData:
    sql_provider: Optional[SqlProvider] = None
    sql_config_path: Optional[str] = None
    sql_result: Optional[List] = None
    sql_result: Optional[Dict[str, List[Dict[str, Any]]]] = None
    # "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
    def __init__(self, **kwargs):
        print(f"self.sql_result: {self.sql_result}")
        super().__init__(**kwargs)
        try:
                
            if 'sql_provider' in kwargs:
                self.sql_provider = kwargs.get('sql_provider')
                self.logger.info("使用传入的 sql_provider")
            else:
                if 'sql_config_path' in kwargs:
                    self.sql_config_path = kwargs.get('sql_config_path')
                    self.logger.info(f"使用传入的配置路径: {self.sql_config_path}")
                    self.sql_provider = SqlProvider(model=SleepIndices, sql_config_path=self.sql_config_path)
                else:
                    default_config_path = '/work/ai/WHOAMI/whoami/scripts/disease_predict/sql_config.yaml'
                    self.logger.info(f"使用默认配置路径: {default_config_path}")
                    self.sql_provider = SqlProvider(model=SleepIndices, sql_config_path=default_config_path)
            
            # 获取数据
            self.logger.info("开始获取睡眠数据...")
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            self.logger.info(f"查询日期范围: {start_date} 至 {end_date}")
            
            result = self.sql_provider.get_record_by_condition(
                condition={},
                exclude_fields=[
                    'health_advice',
                    'sleep_stage_image_x_y',
                    'body_move_image_x_y',
                    'breath_exception_image_sixty_x_y',
                    'heart_bpm_image_x_y',
                    'breath_bpm_image_x_y',
                    'breath_exception_image_x_y',
                    'deep_sleep_second',
                    'total_num_second',
                    'total_num_second_on_bed',
                    'sleep_second',
                    'deep_sleep_second',
                    'waking_second',
                    'to_sleep_second',
                    'leave_bed_total_second',
                    'save_file_path',
                    'creator',
                    'create_time',
                    'updater',
                    'update_time',
                    'deleted',
                    'tenant_id',
                    'id'
                ],
                date_range={"date_field": "query_date", "start_date": start_date, "end_date": end_date}
            )
            
            # 转换键为中文
            self.logger.info(f"获取到 {len(result)} 条记录，正在转换键...")
            field_names_and_descriptions = self.sql_provider.get_field_names_and_descriptions()
            result_list = [self.convert_keys_to_chinese(field_names_and_descriptions, item) for item in result]

            # 按设备分组
            self.sql_result = self.group_by_device_sn(result_list)
            self.logger.info(f"数据分组完成，共有 {len(self.sql_result)} 个设备")
        except Exception as e:
            self.logger.error(f"获取SQL数据失败: {str(e)}")
            self.sql_result = {}
            raise ValueError(f"获取SQL数据失败: {str(e)}") from e
        
        
    @property
    def sql_data(self):
        return self.sql_result
        
        
    def group_by_device_sn(self, data_list):
        result = {}
    
        for item in data_list:
            device_sn = item.get('设备编号')
            if device_sn:
                if device_sn not in result:
                    result[device_sn] = []
                result[device_sn].append(item)
        
        return result

    def convert_keys_to_chinese(self, field_to_chinese_dict, data_dict):
            """
            将数据字典的键从字段名转换为中文名
            
            参数:
            field_to_chinese_dict (dict): 字段名到中文名的映射字典
            data_dict (dict): 需要转换的数据字典，键为字段名，值为字段值
            
            返回:
            dict: 转换后的字典，键为中文名，值为字段值
            """
            # 创建新字典以存储转换后的结果
            converted_dict = {}
            
            # 遍历原始数据字典
            for field, value in data_dict.items():
                # 如果字段在映射字典中存在，使用中文名作为新键
                if field in field_to_chinese_dict:
                    chinese_key = field_to_chinese_dict[field]
                    converted_dict[chinese_key] = value
                else:
                    # 如果字段在映射字典中不存在，保留原始字段名
                    converted_dict[field] = value
            
            # 清空原始字典并用新值更新它
            data_dict.clear()
            data_dict.update(converted_dict)
            
            return data_dict

    
    async def execute(
        self, 
        device_sn: Optional[str] = None
    ) -> str:
        """执行查询并返回数据"""
        if device_sn is not None and device_sn in self.sql_result:
            return {device_sn: self.sql_result[device_sn]}
        return self.sql_result