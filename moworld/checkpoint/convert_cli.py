#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : convert_cli.py
@Time    : 2025/01/12
@Desc    : 权重转换命令行入口
"""
import jsonargparse

from checkpoint.common.converter import Commandable


def main():
    import os
    os.environ['JSONARGPARSE_DEPRECATION_WARNINGS'] = 'off'
    # Allow docstring (including field descriptions) to be parsed as the command-line help documentation.
    # When customizing a converter, you need to inherit from Converter and add it to __init__.py.
    jsonargparse.set_parsing_settings(docstring_parse_attribute_docstrings=True)
    jsonargparse.auto_cli(Commandable.subclasses, as_positional=False)


if __name__ == "__main__":
    main()
