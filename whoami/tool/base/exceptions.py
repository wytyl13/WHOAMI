#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:55:09
@Author : weiyutao
@File : exceptions.py
"""

from whoami.tool.base.base_tool import BaseTool


class BaseException(Exception):
    
    def _run(self, *args, **kwargs):
        raise NotImplementedError

    