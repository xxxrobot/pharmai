# PharmaAI 日志配置指南

本文档介绍如何在PharmaAI项目中使用标准日志框架。

## 概述

PharmaAI使用Python标准库`logging`模块进行日志记录。所有模块都通过`logging.getLogger(__name__)`获取logger实例。

## 基本用法

### 1. 在模块中使用日志

```python
import logging

# 获取logger实例
logger = logging.getLogger(__name__)

# 不同级别的日志
logger.debug("调试信息 - 用于开发调试")
logger.info("一般信息 - 正常运行状态")
logger.warning("警告信息 - 需要注意但不影响运行")
logger.error("错误信息 - 功能异常")
logger.critical("严重错误 - 系统可能无法继续")
```

### 2. 通过Settings配置日志

```python
from pharmaai.core.config import get_settings

# 获取配置并初始化日志
settings = get_settings()
settings.configure_logging()

# 现在所有模块的日志都会按照配置输出
```

### 3. 环境变量配置

通过环境变量控制日志级别：

```bash
# Linux/Mac
export PHARMAAI_LOG_LEVEL=DEBUG

# Windows
set PHARMAAI_LOG_LEVEL=DEBUG
```

可选的日志级别：
- `DEBUG` - 详细调试信息
- `INFO` - 一般信息（默认）
- `WARNING` - 警告信息
- `ERROR` - 错误信息
- `CRITICAL` - 严重错误

## 日志输出位置

默认情况下，日志会输出到两个位置：

1. **控制台** - 实时查看运行状态
2. **文件** - 保存到 `./pharmaai/pharmaai.log`

## 示例代码

### 完整示例

```python
import logging
from pharmaai.core.config import get_settings
from pharmaai.core.utils import MorganFingerprintGenerator, calculate_molecular_features

# 配置日志
settings = get_settings()
settings.configure_logging()

# 获取logger
logger = logging.getLogger(__name__)

# 主程序
logger.info("开始处理分子数据")

try:
    # 生成分子指纹
    fp_gen = MorganFingerprintGenerator.get_instance()
    logger.debug(f"指纹生成器初始化完成: radius={fp_gen.radius}, size={fp_gen.fp_size}")
    
    # 计算分子特征
    smiles = "CCO"  # 乙醇
    features = calculate_molecular_features(smiles)
    logger.info(f"成功计算分子特征: MW={features['MW']:.2f}")
    
except Exception as e:
    logger.error(f"处理过程中发生错误: {e}", exc_info=True)

logger.info("处理完成")
```

### 在自定义预测器中使用

```python
import logging
from pharmaai.core.base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)

class MyPredictor(BasePredictor):
    def __init__(self):
        super().__init__("my_predictor", "1.0.0")
        logger.info(f"初始化预测器: {self.model_name}")
    
    def predict(self, mol):
        logger.debug(f"开始预测分子: {mol}")
        # ... 预测逻辑 ...
        logger.info("预测完成")
        return PredictionResult(value=0.5, confidence=0.9)
```

## 日志格式

默认日志格式：
```
2024-01-15 10:30:45,123 - module_name - INFO - 消息内容
```

格式说明：
- `%(asctime)s` - 时间戳
- `%(name)s` - logger名称（模块名）
- `%(levelname)s` - 日志级别
- `%(message)s` - 日志消息

## 高级配置

### 自定义日志配置

```python
import logging

# 自定义日志配置
def setup_custom_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler('custom.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

setup_custom_logging()
```

### 过滤特定模块的日志

```python
import logging

# 只查看core模块的日志
core_logger = logging.getLogger('pharmaai.core')
core_logger.setLevel(logging.DEBUG)

# 关闭utils模块的调试日志
utils_logger = logging.getLogger('pharmaai.core.utils')
utils_logger.setLevel(logging.WARNING)
```

## 最佳实践

1. **使用适当的日志级别**
   - DEBUG: 详细的调试信息，仅在开发时使用
   - INFO: 重要的运行状态信息
   - WARNING: 需要注意的异常情况
   - ERROR: 功能错误，需要处理
   - CRITICAL: 系统级错误

2. **包含上下文信息**
   ```python
   # 好的做法
   logger.info(f"处理分子: {smiles}, 分子量: {mw}")
   
   # 避免
   logger.info("处理完成")  # 缺少上下文
   ```

3. **使用exc_info记录异常**
   ```python
   try:
       result = risky_operation()
   except Exception as e:
       logger.error(f"操作失败: {e}", exc_info=True)
   ```

4. **避免在循环中记录过多日志**
   ```python
   # 避免在大量迭代中记录日志
   for i, mol in enumerate(molecules):
       if i % 100 == 0:  # 每100个记录一次
           logger.info(f"已处理 {i}/{len(molecules)} 个分子")
       process(mol)
   ```

## 相关文件

- `pharmaai/core/config.py` - 日志配置
- `pharmaai/core/utils.py` - 日志使用示例
- `pharmaai/core/base_predictor.py` - 预测器中的日志
