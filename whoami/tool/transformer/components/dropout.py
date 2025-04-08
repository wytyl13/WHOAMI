#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:41:53
@Author : weiyutao
@File : dropout.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dropout(nn.Dropout):
    # 注意在评估模式下dropout不生效，仅在train模式下生效，因为dropout仅仅是为了在训练阶段防治过拟合
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The default value of inplace is False.
        Unless you are really under memeory pressure, using the default value 
        of False is a safer choice.
        """
        
        output = input if self.p == 0.0 else F.dropout(input, self.p, self.training, self.inplace)
        return output