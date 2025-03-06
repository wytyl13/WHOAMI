from whoami.provider.sql_provider import SqlProvider
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Type,
    Any,
    List
)


from whoami.provider.base_provider import BaseProvider
from whoami.configs.sql_config import SqlConfig
from whoami.provider.base_ import ModelType
from whoami.tool.disease_predict.sx_disease_predict import SxDiseasePredict

class DiseasePredict(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_provider: Optional[SqlProvider] = None
    sick_bed_id: Optional[str] = None
    model: Type[ModelType] = None
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        sql_provider: Optional[SqlProvider] = None,
        sick_bed_id: Optional[str] = None,
        model: Type[ModelType] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path=sql_config_path, sql_config=sql_config, sql_provider=sql_provider, model=model)    
        
        
    def _init_param(self, sql_config_path, sql_config, sql_provider, model):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_provider = sql_provider
        self.model = model
        self.logger.info(self.sql_config_path)
        if self.sql_config_path is None and self.sql_config is None and self.sql_provider is None:
            raise ValueError('sql_config_path, sql_config, sql_provider must not be none!')
        if self.model is None:
            raise ValueError('model must not be null!')
        
        if self.sql_provider is None:
            self.sql_provider = SqlProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, model=self.model)
    
    def _run(self, dept_id, sick_bed_id):

        sql_provider = SqlProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, model=SxDiseasePredict)
        fileds_description = sql_provider.get_field_names_and_descriptions()
        keys = list(fileds_description.keys())
        indices_to_delete = [0, 1, 9, 10, 11]
        for index in sorted(indices_to_delete, reverse=True):
            del fileds_description[keys[index]]
        try:
            result = self.sql_provider.get_record_by_condition({"dept_id": dept_id, "bed_id": sick_bed_id}, fields=["device_sn", "family_id", "room_id"])
            result__ = {}
            indicator = []
            probability_distribution = []
            for key, value in fileds_description.items():
                indicator.append({"name": value, "max": 1})
                result__["indicator"] = indicator
                result__["probability_distribution"] = probability_distribution
            if result:
                device_sn = result[0]["device_sn"]
            else:
                return result__
        except Exception as e:
            error_info = f"fail to check the device_sn based on the sick_bed_id!{str(e)}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e
        result = sql_provider.get_record_by_condition({"device_sn": device_sn}, exclude_fields=["device_sn", "creator", "id", "create_time", "deleted"])
        result_ = {}
        if result:
            indicator = []
            probability_distribution = []
            for key, value in result[0].items():
                indicator.append({"name": fileds_description[key], "max": 1})
                probability_distribution.append(value)
            result_["indicator"] = indicator
            result_["probability_distribution"] = probability_distribution
            return result_
        return result__
        
