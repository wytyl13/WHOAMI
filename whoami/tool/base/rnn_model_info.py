#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 11:09
@Author  : weiyutao
@File    : model_info.py
"""
from whoami.provider.base_ import ModelType
from whoami.tool.base.base_tool import BaseTool
from whoami.neural_network.rnn.model import LSTM
from whoami.neural_network.rnn.model import RNNBase
from whoami.neural_network.rnn.auto_encoders import RecurrentAutoencoder

class RNNModelInfo(BaseTool):

    model_path: str = None
    model_type_class: RNNBase = None
    classes: list = None
    conf: float = None

    def __init__(self, 
            model_path: str = None, 
            model_type_class: ModelType = None,
            classes: list = None,
            conf: float = None
        ):
        super().__init__()
        self.model_path = model_path
        self.model_type_class = model_type_class
        self.classes = classes
        self.conf = conf
    def init_model(self):
        if self.model_type_class == LSTM:
            return LSTM(28, 128, 1, dropout=0.2, bidirectional=False)
        raise ValueError(f'Invalid model type! model_type_class: {self.model_type_class}')
    def _run(self, *args, **kwargs):
        pass