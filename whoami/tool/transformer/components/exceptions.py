#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:58:05
@Author : weiyutao
@File : exceptions.py
"""

class BaseException(Exception):
    
    def _run(self, *args, **kwargs):
        raise NotImplementedError


class TLMoError(BaseException):
    """
    Base class for all custom TLMo exceptions.
    """


class TLMoConfigurationError(TLMoError):
    """
    An error with a configuration file.
    """

    