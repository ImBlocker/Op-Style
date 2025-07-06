import sys
import os

# 获取项目根目录路径（根据你的实际路径调整）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 然后再导入其他模块
from modules.Xfeatmodel import *