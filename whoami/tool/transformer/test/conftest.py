import sys
import os

# 添加项目根目录到 Python 路径
# 仅仅为了pytest测试做准备，因为test目录下所有测试文件的导入都使用的是whoami包开头
# 而我们在WHOAMI目录下（也就是whoami包的父目录下）执行测试脚本，需要拼接绝对路径才可以导入whoami开头的文件
# 这个文件pytest自动读取,我们可以直接在whoami包的上级目录下使用 pytest -s whoami/tool/transformer/test/model_config_test.py 执行某一个测试文件
# 类似于在whoami包下使用 python -m whoami.tool.transformer.test.model_config_test，这个指令实际上也是添加了whoami包的父路径到python路径
sys.path.insert(0, "/work/ai/WHOAMI")