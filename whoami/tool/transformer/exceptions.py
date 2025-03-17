#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:58:05
@Author : weiyutao
@File : exceptions.py
"""


from whoami.tool.base.exceptions import BaseException



class TLMoError(BaseException):
    """
    Base class for all custom TLMo exceptions.
    """


class TLMoCOnfigurationError(TLMoError):
    """
    An error with a configuration file.
    """

    