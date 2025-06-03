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
import numpy as np
from scipy import stats
import math

from whoami.provider.base_provider import BaseProvider
from whoami.configs.sql_config import SqlConfig
from whoami.provider.base_ import ModelType

class ThresholdValue(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_provider: Optional[SqlProvider] = None
    model: Type[ModelType] = None
    device_sn: Optional[str] = None
    
    
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        sql_provider: Optional[SqlProvider] = None,
        model: Type[ModelType] = None,
        device_sn: Optional[str] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path=sql_config_path, sql_config=sql_config, sql_provider=sql_provider, model=model, device_sn=device_sn)  
    
    
    def _init_param(self, sql_config_path, sql_config, sql_provider, model, device_sn):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_provider = sql_provider
        self.model = model
        self.device_sn = device_sn
        self.logger.info(self.sql_config_path)
        if self.sql_config_path is None and self.sql_config is None and self.sql_provider is None:
            raise ValueError('sql_config_path, sql_config, sql_provider must not be none!')
        if self.model is None:
            raise ValueError('model must not be null!')
        
        if self.sql_provider is None:
            self.sql_provider = SqlProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, model=self.model)
    
    
    def calculate_thresholds(self, data, method='percentile', 
                        window_days=7,
                        lower_percentile=1, 
                        upper_percentile=99,
                        zscore_threshold=3,
                        min_samples=1000):
        """
        Calculate thresholds for physiological data using various statistical methods.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Time series data of vital signs
        method : str, optional
            Method to calculate thresholds:
            - 'percentile': Uses percentile-based thresholds
            - 'zscore': Uses mean ± n*std thresholds
            - 'mad': Uses Median Absolute Deviation
        window_days : int, optional
            Number of days of data to use for calculation
        lower_percentile : float, optional
            Lower percentile threshold (for percentile method)
        upper_percentile : float, optional
            Upper percentile threshold (for percentile method)
        zscore_threshold : float, optional
            Number of standard deviations (for zscore method)
        min_samples : int, optional
            Minimum number of samples required for calculation
            
        Returns:
        --------
        dict
            Dictionary containing lower and upper thresholds
        """
        
        if len(data) < min_samples:
            raise ValueError(f"Insufficient data: needs at least {min_samples} samples")
            
        # Remove extreme outliers (values beyond 5 standard deviations)
        mean = np.mean(data)
        std = np.std(data)
        cleaned_data = data[np.abs(data - mean) <= 5 * std]
        
        if method == 'percentile':
            lower = np.percentile(cleaned_data, lower_percentile)
            upper = np.percentile(cleaned_data, upper_percentile)
            
        elif method == 'zscore':
            mean = np.mean(cleaned_data)
            std = np.std(cleaned_data)
            lower = mean - zscore_threshold * std
            upper = mean + zscore_threshold * std
            
        elif method == 'mad':
            # Median Absolute Deviation method
            median = np.median(cleaned_data)
            mad = stats.median_abs_deviation(cleaned_data)
            lower = median - zscore_threshold * mad
            upper = median + zscore_threshold * mad
            
        else:
            raise ValueError("Invalid method specified")
            
        # Calculate additional statistics
        stats_info = {
            'mean': np.mean(cleaned_data),
            'median': np.median(cleaned_data),
            'std': np.std(cleaned_data),
            'data_points': len(cleaned_data)
        }
        
        return {
            'lower_threshold': lower,
            'upper_threshold': upper,
            'statistics': stats_info,
            'method': method
        }

    
    def calculate_adaptive_thresholds(self, data, min_value, max_value):
        """
        计算考虑实际数据范围的自适应阈值
        """
        # 计算统计阈值
        stats_thresholds = self.calculate_thresholds(
            data=data,
            method='percentile',
            lower_percentile=0.01,
            upper_percentile=99.95
        )
        
        # 动态调整阈值
        lower_threshold = min(
            stats_thresholds['lower_threshold'],
            min_value * 1.1  # 允许比最小值低10%
        )
        
        upper_threshold = max(
            stats_thresholds['upper_threshold'],
            max_value * 0.9  # 允许比最大值高10%
        )
        
        return {
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'statistics': stats_thresholds['statistics'],
            'method': 'adaptive_percentile'
        }


    def _run(self, query_date: Optional[str] = None, device_sn: Optional[str] = None, breath_bpm: Optional[np.array] = None, heart_bpm: Optional[np.array] = None):
        device_sn = self.device_sn if device_sn is None else device_sn
        
        if device_sn is None:
            raise ValueError('device_sn must not be null!')
        breath_bpm_low_default = 7.00
        breath_bpm_high_default = 36.00
        heart_bpm_low_default = 45.00
        heart_bpm_high_default = 140.00
        if device_sn == "default_config":
            return {
                "device_sn": device_sn,
                "query_date": query_date,
                "breath_bpm_low": 7.00, 
                "breath_bpm_high": 36.00, 
                "heart_bpm_low": 45.00, 
                "heart_bpm_high": 140.00,
                "alarm_time_interval": 10,
                "min_breath_bpm": 0,
                "max_breath_bpm": 0,
                "min_heart_bpm": 0,
                "max_heart_bpm": 0,
                "error_info": "default config for other device_sn!"
            }

        if query_date is None:
            raise ValueError('query_date must not be null!')
        cleaned_breath_bpm = None if breath_bpm is None else breath_bpm
        cleaned_heart_bpm = None if heart_bpm is None else heart_bpm
        
        if cleaned_breath_bpm is None or cleaned_heart_bpm is None:
            results = self.sql_provider.get_record_by_condition({"device_sn": device_sn, "create_date": query_date}, fields=['breath_bpm', 'heart_bpm'])
            breath_bpm = [item['breath_bpm'] for item in results]
            heart_bpm = [item['heart_bpm'] for item in results]
            breath_arr = np.array(breath_bpm)
            heart_arr = np.array(heart_bpm)
            mask = (breath_arr != 0) & (heart_arr != 0)
            cleaned_breath_bpm = breath_arr[mask]
            cleaned_heart_bpm = heart_arr[mask]
        
        if len(cleaned_breath_bpm) < 10000:
            return {
            "device_sn": device_sn,
            "query_date": query_date,
            "breath_bpm_low": 0.00, 
            "breath_bpm_high": 100.00, 
            "heart_bpm_low": 0.00, 
            "heart_bpm_high": 200.00,
            "alarm_time_interval": 10,
            "min_breath_bpm": 0,
            "max_breath_bpm": 0,
            "min_heart_bpm": 0,
            "max_heart_bpm": 0,
            "error_info": f"in bed data is less than 3 hours! device_sn: {device_sn}, query_date: {query_date}"
        }
        
        
        max_breath_bpm_value = np.max(cleaned_breath_bpm)
        max_heart_bpm_value = np.max(cleaned_heart_bpm)
        min_breath_bpm_value = np.min(cleaned_breath_bpm)
        min_heart_bpm_value = np.min(cleaned_heart_bpm)
        thresholds_breath = self.calculate_adaptive_thresholds(cleaned_breath_bpm, min_breath_bpm_value, max_breath_bpm_value)
        thresholds_heart = self.calculate_adaptive_thresholds(cleaned_heart_bpm, min_heart_bpm_value, max_heart_bpm_value)
        count_upper_breath = len([x for x in cleaned_breath_bpm if x > thresholds_breath["upper_threshold"]])
        count_lower_breath = len([x for x in cleaned_breath_bpm if x < thresholds_breath["lower_threshold"]])
        count_upper_heart = len([x for x in cleaned_heart_bpm if x > thresholds_heart["upper_threshold"]])
        count_lower_heart = len([x for x in cleaned_heart_bpm if x < thresholds_heart["lower_threshold"]])
        return {
            "device_sn": device_sn,
            "query_date": query_date,
            "breath_bpm_low": math.floor(min(np.round(thresholds_breath["lower_threshold"], 6), breath_bpm_low_default)), 
            "breath_bpm_high": math.ceil(max(np.round(thresholds_breath["upper_threshold"], 6), breath_bpm_high_default)), 
            "heart_bpm_low": math.floor(min(np.round(thresholds_heart["lower_threshold"], 6), heart_bpm_low_default)), 
            "heart_bpm_high": math.ceil(max(np.round(thresholds_heart["upper_threshold"], 6), heart_bpm_high_default)),
            "alarm_time_interval": 10,
            "min_breath_bpm": min_breath_bpm_value,
            "max_breath_bpm": max_breath_bpm_value,
            "min_heart_bpm": min_heart_bpm_value,
            "max_heart_bpm": max_heart_bpm_value,
            "error_info": ""
        }
        