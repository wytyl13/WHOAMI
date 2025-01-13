#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/07 11:15
@Author  : weiyutao
@File    : runnable.py
"""

from abc import ABC, abstractmethod
from typing import (
    TypeVar, 
    Generic,
    Optional,
    Any
)

from whoami.core.utils import (
    Input, 
    Output
)

class Runable(Generic[Input, Output], ABC):
    
      
        """Transform a single input into an output. Override to implement.

        Args:
            input: The input to the Runnable.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details.

        Returns:
            The output of the Runnable.
        """