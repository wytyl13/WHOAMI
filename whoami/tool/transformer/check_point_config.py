#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/10 22:38:21
@Author : weiyutao
@File : check_point_config.py
"""
from enum import Enum
from enum import IntEnum
from whoami.utils.utils import StrEnum


class ActivationCheckpointStrategy(StrEnum):
    
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    
    """
    Checkpoint one in two transformer layers.
    """
    
    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layer.
    """
    
    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    one_in_eight = "one_in_eight"
    """
    Checkpoint one in eight transformer layers.
    """
    
    two_in_three = "two_in_three"
    """
    Checkpoint two in three transformer layers.
    """
    
    three_in_four = "three_in_four"
    """
    Checkpoint three in four transformer layers.
    """
    
    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """
    
    
    
    
    
    

