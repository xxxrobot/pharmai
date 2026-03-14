#!/usr/bin/env python3
"""
毒性预测模块 (兼容层)
已弃用 - 请使用新的 pharmaai.predictors.toxicity 模块

此模块已迁移到统一架构中，此文件仅为向后兼容而保留
新代码应导入: from pharmaai.predictors.toxicity import ToxicityPredictor
"""

import warnings
warnings.warn(
    "pharma_toxicity_prediction.py 已弃用，请使用 pharmaai.predictors.toxicity",
    DeprecationWarning,
    stacklevel=2
)

import sys
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

# 导入新模块的预测器类以保持 API 兼容
try:
    from pharmaai.predictors.toxicity import ToxicityPredictor as NewToxicityPredictor
    from pharmaai.predictors.toxicity import ToxicityType
    from pharmaai.predictors.toxicity import create_toxicity_predictor
except ImportError as e:
    warnings.warn(f"无法导入新模块: {e}", ImportWarning)
    # 定义兼容的占位符类，防止导入失败
    class NewToxicityPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("ToxicityPredictor 需要安装新模块。请使用 pip install -e .")
    
    class ToxicityType:
        HERG = "hERG"
        HEPATOTOXICITY = "Hepatotoxicity"
        AMES = "Ames"
    
    def create_toxicity_predictor(*args, **kwargs):
        raise ImportError("create_toxicity_predictor 需要安装新模块。请使用 pip install -e .")


# ===== 兼容层: 原始 ToxicityPrediction 类 =====

class ToxicityPrediction:
    """
    毒性预测兼容类
    
    此类为旧代码提供向后兼容性，实际功能委托给新的 ToxicityPredictor
    """
    
    def __init__(self, toxicity_type='hERG', model_path=None):
        """
        初始化毒性预测器 (兼容版本)
        
        Args:
            toxicity_type: 毒性类型 ('hERG', 'Hepatotoxicity', 'Ames')
            model_path: 模型文件路径（可选）
        """
        warnings.warn(
            "ToxicityPrediction 已弃用，请使用 pharmaai.predictors.toxicity.ToxicityPredictor",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.toxicity_type = toxicity_type
        
        # 映射旧类型到新枚举
        toxicity_map = {
            'hERG': ToxicityType.HERG,
            'Hepatotoxicity': ToxicityType.HEPATOTOXICITY,
            'Ames': ToxicityType.AMES
        }
        
        mapped_type = toxicity_map.get(toxicity_type, ToxicityType.HERG)
        
        # 创建新的预测器实例
        self._predictor = NewToxicityPredictor(toxicity_type=mapped_type)
        
        # 加载模型文件（如果提供）
        if model_path:
            self._predictor.load_model(model_path)
    
    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """
        训练毒性预测模型 (兼容方法)
        
        Args:
            df: 包含 'smiles' 和毒性标签列的 DataFrame
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        return self._predictor.train(df, test_size=test_size, random_state=random_state)
    
    def predict(self, smiles: str) -> Dict[str, Any]:
        """
        预测单个分子的毒性 (兼容方法)
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            包含预测结果的字典
        """
        result = self._predictor.predict(smiles)
        
        # 转换为旧格式
        return {
            'smiles': smiles,
            'is_toxic': result.value,
            'confidence': result.confidence,
            'toxicity_type': self.toxicity_type,
            'metadata': result.metadata
        }
    
    def batch_predict(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测 (兼容方法)
        
        Args:
            smiles_list: SMILES 字符串列表
            
        Returns:
            预测结果列表
        """
        results = self._predictor.batch_predict(smiles_list)
        
        # 转换为旧格式
        output = []
        for smiles, result in zip(smiles_list, results):
            output.append({
                'smiles': smiles,
                'is_toxic': result.value,
                'confidence': result.confidence,
                'toxicity_type': self.toxicity_type,
                'metadata': result.metadata
            })
        return output
    
    def save_model(self, filepath: str) -> bool:
        """
        保存模型 (兼容方法)
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        return self._predictor.save_model(filepath)
    
    def load_model(self, filepath: str) -> bool:
        """
        加载模型 (兼容方法)
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            是否加载成功
        """
        return self._predictor.load_model(filepath)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性 (兼容方法)
        
        Returns:
            特征重要性字典
        """
        if hasattr(self._predictor, 'feature_importance'):
            return self._predictor.feature_importance
        return {}
    
    def get_toxicity_alerts(self, smiles: str) -> Dict[str, bool]:
        """
        获取毒性警示结构 (兼容方法)
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            警示结构字典
        """
        result = self._predictor.predict(smiles)
        alerts = result.metadata.get('alert_features', {})
        return alerts


# ===== 兼容层: 原有全局函数 =====

def load_toxicity_model(toxicity_type='hERG', model_path=None) -> ToxicityPrediction:
    """
    加载毒性模型 (兼容函数)
    
    Args:
        toxicity_type: 毒性类型
        model_path: 模型文件路径
        
    Returns:
        ToxicityPrediction 实例
    """
    warnings.warn(
        "load_toxicity_model 已弃用，请使用 pharmaai.predictors.toxicity.create_toxicity_predictor",
        DeprecationWarning,
        stacklevel=2
    )
    return ToxicityPrediction(toxicity_type=toxicity_type, model_path=model_path)


def predict_toxicity(smiles: str, toxicity_type='hERG', model_path=None) -> Dict[str, Any]:
    """
    预测毒性 (兼容函数)
    
    Args:
        smiles: SMILES 字符串
        toxicity_type: 毒性类型
        model_path: 模型文件路径
        
    Returns:
        包含预测结果的字典
    """
    warnings.warn(
        "predict_toxicity 已弃用，请使用 pharmaai.predictors.toxicity.ToxicityPredictor",
        DeprecationWarning,
        stacklevel=2
    )
    
    predictor = ToxicityPrediction(toxicity_type=toxicity_type, model_path=model_path)
    return predictor.predict(smiles)


def batch_predict_toxicity(smiles_list: List[str], toxicity_type='hERG', model_path=None) -> List[Dict[str, Any]]:
    """
    批量预测毒性 (兼容函数)
    
    Args:
        smiles_list: SMILES 字符串列表
        toxicity_type: 毒性类型
        model_path: 模型文件路径
        
    Returns:
        预测结果列表
    """
    warnings.warn(
        "batch_predict_toxicity 已弃用，请使用 pharmaai.predictors.toxicity.ToxicityPredictor.batch_predict",
        DeprecationWarning,
        stacklevel=2
    )
    
    predictor = ToxicityPrediction(toxicity_type=toxicity_type, model_path=model_path)
    return predictor.batch_predict(smiles_list)


# ===== 保持原有模块属性 =====

__all__ = [
    'ToxicityPrediction',
    'load_toxicity_model',
    'predict_toxicity',
    'batch_predict_toxicity',
    'ToxicityPredictor',  # 新预测器类（如果导入成功）
    'ToxicityType',       # 新枚举类型（如果导入成功）
    'create_toxicity_predictor'  # 新创建函数（如果导入成功）
]

# 如果新预测器导入成功，将其导出
try:
    ToxicityPredictor = NewToxicityPredictor
except NameError:
    pass

if __name__ == '__main__':
    # 兼容性测试
    print("毒性预测模块兼容层已加载")
    print("注意: 此模块已弃用，请迁移到 pharmaai.predictors.toxicity")