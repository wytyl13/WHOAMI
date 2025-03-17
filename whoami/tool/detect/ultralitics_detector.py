#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 11:26
@Author  : weiyutao
@File    : yolo_detector.py
"""
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    List,
    overload,
)
from pydantic import BaseModel, model_validator, ValidationError
import torch
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO, RTDETR

from whoami.tool.detect.detector import Detector

class UltraliticsDetector(Detector):
    
    def __init__(
        self, 
        name: Optional[str] = None,
        model: Optional[Union[YOLO, RTDETR]] = None, 
        model_path: Optional[str] = None,
        class_list: Optional[list[int]] = None,
        conf: Optional[float] = None,
    ):
        super().__init__(name=name, model=model, model_path=model_path, class_list=class_list, conf=conf)
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
        
    def tostring(self):
        return {
            "name": self.name,
            "model_path": self.model_path, 
            "class_list": self.class_list, 
            "conf": self.conf, 
        }    
    
    @model_validator(mode="after")
    @classmethod
    def valid_model_model_path(cls, data: any):
        if data.model is None and data.model_path is None:
            raise ValueError("Either model or model_path must be provided!")
        if data.model_path:
            data.model = RTDETR(data.model_path) if 'detr' in data.model_path else YOLO(data.model_path)
        return data
    
    
    @model_validator(mode="after")
    @classmethod
    def valid_model_model_path_(cls, data: any):
        """valid model and model path and init them implemented by inherited class"""
        return data
    
    def set_model(self, model_path):
        """set model used model_path implemented by inherited class"""
        try:
            self.model = RTDETR(model_path) if 'detr' in model_path else YOLO(model_path)
        except Exception as e:
            raise ValueError(f"invalid model path {model_path}") from e
    
    def predict(
        self, 
        image: Optional[Union[str, np.ndarray, List[Union[str, np.ndarray]]]] = None,
    ):
        if image is None:
            raise ValueError("image must not be None!")
        
        # Check if it's a batch (list) of images
        is_batch = isinstance(image, list)

        # Convert single image to list for unified processing
        images = image if is_batch else [image]

        # Validate all images in the batch
        for i, img in enumerate(images):
            if not isinstance(img, (str, np.ndarray)):
                raise ValueError(f"Image at index {i} must be str or np.ndarray!")
            
            if isinstance(img, str) and not Path(img).exists():
                raise ValueError(f"Image path {img} at index {i} does not exist!")

        try:
            with torch.no_grad():
                results = self.model.predict(
                    source=images, 
                    classes=self.class_list, 
                    conf=self.conf, 
                    verbose=False
                )
        except Exception as e:
            raise RuntimeError(f"Failed to predict {'batch' if is_batch else 'single'} image! {str(e)}") from e
        
        return results

    def predict_bake(
            self, 
            image: Optional[Union[str, np.ndarray]] = None):
        if image is None:
            raise ValueError("image must not be None!")
        
        if not isinstance(image, (str, np.ndarray)):
            raise ValueError("image must be str or np.ndarray!")
        
        if isinstance(image, str) and not Path(image).exists():
            raise ValueError(f"image path {image} is not exists!")
        
        try:
            with torch.no_grad():
                results = self.model.predict(source=image, classes=self.class_list, conf=self.conf, verbose=False)
        except Exception as e:
            raise RuntimeError(f"fail to predict one image! {str(e)}") from e
        return results

    # def predict(
    #         self, 
    #         images):
        
    #     if images is None or not images:
    #         raise ValueError("images must not be None!")
    #     self.logger.info("whoami")
    #     try:
    #         with torch.no_grad():
    #             if not isinstance(images, list):
    #                 self.logger.info("single images----------------------------")
    #                 results = self._predict(source=images, classes=self.class_list, conf=self.conf)
    #             else:
    #                 self.logger.info(f"batch images-{len(images)}---------------------------")
    #                 results = self.model.predict(source=images, classes=self.class_list, conf=self.conf)
    #     except Exception as e:
    #         raise RuntimeError(f"fail to predict one image! {str(e)}") from e
    #     return results
        

    
    