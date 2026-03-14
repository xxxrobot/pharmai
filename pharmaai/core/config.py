"""
PharmaAI 配置管理模块

使用pydantic-settings管理配置，支持环境变量覆盖
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # 回退到简单的配置类
    class BaseSettings:
        pass

logger = logging.getLogger(__name__)


class Settings:
    """
    PharmaAI 配置类
    
    支持从环境变量加载配置，配置项包括：
    - 模型路径
    - 指纹参数
    - 日志级别
    - 数据目录
    
    Environment Variables:
        PHARMAAI_MODEL_PATH: 模型存储路径
        PHARMAAI_DATA_PATH: 数据存储路径
        PHARMAAI_LOG_LEVEL: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        PHARMAAI_FINGERPRINT_RADIUS: Morgan指纹半径
        PHARMAAI_FINGERPRINT_SIZE: Morgan指纹大小
    """
    
    # 默认配置
    DEFAULT_MODEL_PATH = "./models"
    DEFAULT_DATA_PATH = "./data"
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_FINGERPRINT_RADIUS = 2
    DEFAULT_FINGERPRINT_SIZE = 2048
    DEFAULT_OUTPUT_DIR = "./pharma_demo"
    
    def __init__(self):
        """初始化配置"""
        self.model_path: str = os.getenv("PHARMAAI_MODEL_PATH", self.DEFAULT_MODEL_PATH)
        self.data_path: str = os.getenv("PHARMAAI_DATA_PATH", self.DEFAULT_DATA_PATH)
        self.log_level: str = os.getenv("PHARMAAI_LOG_LEVEL", self.DEFAULT_LOG_LEVEL)
        self.fingerprint_radius: int = int(os.getenv("PHARMAAI_FINGERPRINT_RADIUS", self.DEFAULT_FINGERPRINT_RADIUS))
        self.fingerprint_size: int = int(os.getenv("PHARMAAI_FINGERPRINT_SIZE", self.DEFAULT_FINGERPRINT_SIZE))
        self.output_dir: str = os.getenv("PHARMAAI_OUTPUT_DIR", self.DEFAULT_OUTPUT_DIR)
        
        # CYP450预测器配置
        self.cyp450_isoforms: List[str] = ["CYP3A4", "CYP2D6", "CYP2C9"]
        self.cyp450_threshold: float = 0.5
        
        # 毒性预测配置
        self.toxicity_models: List[str] = ["hERG", "Hepatotoxicity", "Ames"]
        self.toxicity_threshold: float = 0.5
        
        # ADMET预测配置
        self.admet_properties: List[str] = [
            "Solubility", "Permeability", "Metabolic_Stability",
            "Bioavailability", "Half_Life", "Clearance"
        ]
        
        # 确保目录存在
        self._ensure_directories()
        
        logger.debug(f"Settings loaded: model_path={self.model_path}, log_level={self.log_level}")
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.model_path,
            self.data_path,
            self.output_dir,
            f"{self.output_dir}/models",
            f"{self.output_dir}/results",
            f"{self.output_dir}/data",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        获取特定模型的完整路径
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型文件完整路径
        """
        return os.path.join(self.model_path, f"{model_name}.joblib")
    
    def get_data_path(self, filename: str) -> str:
        """
        获取数据文件的完整路径
        
        Args:
            filename: 数据文件名
            
        Returns:
            数据文件完整路径
        """
        return os.path.join(self.data_path, filename)
    
    def configure_logging(self):
        """
        配置日志系统
        
        使用标准logging模块，配置格式和级别
        """
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, "pharmaai.log"))
            ]
        )
        
        logger.info(f"Logging configured with level {self.log_level}")
    
    def to_dict(self) -> dict:
        """
        将配置转换为字典
        
        Returns:
            配置字典
        """
        return {
            "model_path": self.model_path,
            "data_path": self.data_path,
            "log_level": self.log_level,
            "fingerprint_radius": self.fingerprint_radius,
            "fingerprint_size": self.fingerprint_size,
            "output_dir": self.output_dir,
            "cyp450_isoforms": self.cyp450_isoforms,
            "cyp450_threshold": self.cyp450_threshold,
            "toxicity_models": self.toxicity_models,
            "toxicity_threshold": self.toxicity_threshold,
            "admet_properties": self.admet_properties,
        }
    
    def __repr__(self) -> str:
        return f"Settings(model_path='{self.model_path}', log_level='{self.log_level}')"


# 全局配置实例
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局配置实例 (单例模式)
    
    Returns:
        Settings实例
        
    Example:
        >>> from pharmaai.core.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.model_path)
        ./models
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reset_settings():
    """重置配置实例（主要用于测试）"""
    global _settings_instance
    _settings_instance = None
    logger.debug("Settings reset")


# 如果使用pydantic-settings，提供更高级的配置类
if PYDANTIC_AVAILABLE:
    class PydanticSettings(BaseSettings):
        """
        基于Pydantic的高级配置类
        
        需要安装pydantic-settings: pip install pydantic-settings
        """
        model_path: str = Field(default="./models", env="PHARMAAI_MODEL_PATH")
        data_path: str = Field(default="./data", env="PHARMAAI_DATA_PATH")
        log_level: str = Field(default="INFO", env="PHARMAAI_LOG_LEVEL")
        fingerprint_radius: int = Field(default=2, env="PHARMAAI_FINGERPRINT_RADIUS")
        fingerprint_size: int = Field(default=2048, env="PHARMAAI_FINGERPRINT_SIZE")
        output_dir: str = Field(default="./pharma_demo", env="PHARMAAI_OUTPUT_DIR")
        
        class Config:
            env_prefix = "PHARMAAI_"
            case_sensitive = False
    
    # 导出Pydantic配置类
    __all__ = ["Settings", "get_settings", "reset_settings", "PydanticSettings"]
else:
    __all__ = ["Settings", "get_settings", "reset_settings"]
