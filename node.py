# node.py
import importlib
import os
import sys
from pathlib import Path

# 获取当前文件目录
current_dir = Path(__file__).parent

# 定义要导入的模块名（不包括.py扩展名）
modules_to_import = [
    'RunningHubRhartImageToImageAlioss',
    'YK_Vision_ActionPrompt',
    'YK_Vision_ActionPrompt_v2',
    'oss_random_loader'
]

# 初始化节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 动态导入每个模块并收集节点定义
for module_name in modules_to_import:
    try:
        # 构建模块路径
        module_path = current_dir / f"{module_name}.py"
        
        if module_path.exists():
            # 使用importlib导入模块
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 从模块中提取节点映射
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS'))
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'))
                
    except Exception as e:
        print(f"Warning: Could not import {module_name}: {e}")
        continue

# 验证导入的节点数量
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes")