#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/27 11:13
@Author  : weiyutao
@File    : lstm.py

LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.

Mechanical devices such as engines, vehicles, aircrafts are typically instrumented with numerous
sensors to capture the behavior and health of the machine. However, there are often external factors
or variables which are not captured by sensors leading to time-series which are inherently unpredictable.
For instance, manual controls or unmonitored environmental conditions or load may lead to inherently 
unpredictable time-series. Detecting anomalies in such scenarios becomes challenging using standard
approaches based on mathematical models that rely on stationarity, or prediction models that utilize prediction
errors to detect anomalies. 


Definition of anomalies
Pointwise: A data point is anomalous if this point is distant from other observations according to some specific measurement metrics.
This is used in fine-grained anomaly detection tasks, that need to find out every single anomalous instance, credit card fraud detection,
spam email detection.

Window-based: Sometimes, a data point is apparently normal, but this point, or potentially together with its neighbors violates the overall
periodicity or other character of the time series, we also treat theme as anomaly, which is called window-based anomaly or contextual anomaly.

The long short-term memory networks are a kind of reinforces RNN that are able to remember valuable information in arbitrary time interval.



"""

import torch
import torch.nn as nn
from typing import (
    Optional,
    overload
)

class LSTMAutoEncoder(nn.Module):
    @overload
    def __init__(
        self,
        num_layers,
        hidden_size,
        nb_feature,
        dropout=0,
        device=torch.device('cpu')
    ):
        ...
        
        
    @overload
    def __init__(self, *args, **kwargs):
        ...
    
        
    def __init__(self, *args, **kwargs):
        if 'num_layers' in kwargs:
            self.num_layers = kwargs.pop('num_layers', 3)
        if 'hidden_size' in kwargs:
            self.hidden_size = kwargs.pop('hidden_size', 128)
        if 'nb_feature' in kwargs:
            self.nb_feature = kwargs.pop('nb_feature', 28)    
        if 'dropout' in kwargs:
            self.dropout = kwargs.pop('dropout', 0)
        if 'device' in kwargs:
            self.device = kwargs.pop('device', torch.device('cpu'))
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(self.num_layers, self.hidden_size, self.nb_feature, self.dropout, self.device)
        self.decoder = Decoder(self.num_layers, self.hidden_size, self.nb_feature, self.dropout, self.device)
        
        
    def forward(self, input_seq):
        output = torch.zeros(size=input_seq.shape, dtype=torch.float)
        hidden_cell = self.encoder(input_seq)
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])
        for i in range(input_seq.shape[1] - 1, -1, -1):
            try:
                output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
                input_decoder = output_decoder
                output[:, i, :] = output_decoder[:, 0, :]
            except Exception as e:
                raise ValueError("fail to exec decode forward function") from e
        return output
            
    

class Encoder(nn.Module):
    def __init__(
        self,
        num_layers, 
        hidden_size, 
        nb_feature,
        dropout=0,
        device=torch.device('cpu')
    ):
        super().__init__()
        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
    
    
    def init_hidden(self, batch_size):
        self.hidden_cell = (
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device),
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        )
    
    
    def forward(self, input_seq):
        self.init_hidden(input_seq.shape[0])
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        return self.hidden_cell


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super().__init__()
        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)
    
    
    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell


