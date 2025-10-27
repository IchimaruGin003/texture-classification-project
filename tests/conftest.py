"""
Pytest configuration file - sets up Python path for all tests
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# 移除未使用的KNNTrainer导入，因为我们已经通过测试验证了导入功能
# 这个文件的主要目的是设置Python路径，导入验证已经在测试中完成
