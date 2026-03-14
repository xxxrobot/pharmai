"""
PharmaAI 预测器模块

统一预测器接口，所有预测器都继承自 BasePredictor 基类
"""

__version__ = "0.1.0"
__author__ = "PharmaAI Team"

# 导出所有预测器类
from .cyp450 import CYP450Predictor, create_cyp450_predictor
from .toxicity import ToxicityPredictor, ToxicityType, create_toxicity_predictor
from .admet import ADMETPredictor, ADMETType, create_admet_predictor

__all__ = [
    # CYP450 预测器
    "CYP450Predictor",
    "create_cyp450_predictor",
    
    # 毒性预测器
    "ToxicityPredictor",
    "ToxicityType",
    "create_toxicity_predictor",
    
    # ADMET 预测器
    "ADMETPredictor",
    "ADMETType",
    "create_admet_predictor",
]

# 可用预测器列表
PREDICTOR_CLASSES = {
    'cyp450': CYP450Predictor,
    'toxicity': ToxicityPredictor,
    'admet': ADMETPredictor,
}

# 预测器类型枚举
PREDICTOR_TYPES = {
    'cyp450': ['CYP3A4', 'CYP2D6', 'CYP2C9'],
    'toxicity': ['hERG', 'Hepatotoxicity', 'Ames'],
    'admet': ['Solubility', 'MetabolicStability'],
}

def get_predictor_class(predictor_type: str):
    """
    获取预测器类
    
    Args:
        predictor_type: 预测器类型 ('cyp450', 'toxicity', 'admet')
        
    Returns:
        预测器类
    """
    return PREDICTOR_CLASSES.get(predictor_type)

def create_predictor(predictor_type: str, **kwargs):
    """
    创建预测器实例的便捷函数
    
    Args:
        predictor_type: 预测器类型 ('cyp450', 'toxicity', 'admet')
        **kwargs: 传递给预测器构造函数的参数
        
    Returns:
        预测器实例
    """
    predictor_class = get_predictor_class(predictor_type)
    if predictor_class is None:
        raise ValueError(f"未知的预测器类型: {predictor_type}")
    return predictor_class(**kwargs)