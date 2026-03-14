#!/usr/bin/env python3
"""
ADMET 性质预测模块 (兼容层)
已弃用 - 请使用新的 pharmaai.predictors.admet 模块

此模块已迁移到统一架构中，此文件仅为向后兼容而保留
新代码应导入: from pharmaai.predictors.admet import ADMETPredictor
"""

import warnings
warnings.warn(
    "pharma_admet_prediction.py 已弃用，请使用 pharmaai.predictors.admet",
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
    from pharmaai.predictors.admet import ADMETPredictor as NewADMETPredictor
    from pharmaai.predictors.admet import ADMETType
    from pharmaai.predictors.admet import create_admet_predictor
except ImportError as e:
    warnings.warn(f"无法导入新模块: {e}", ImportWarning)
    # 定义兼容的占位符类，防止导入失败
    class NewADMETPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("ADMETPredictor 需要安装新模块。请使用 pip install -e .")
    
    class ADMETType:
        SOLUBILITY = "Solubility"
        METABOLIC_STABILITY = "MetabolicStability"
    
    def create_admet_predictor(*args, **kwargs):
        raise ImportError("create_admet_predictor 需要安装新模块。请使用 pip install -e .")


# ===== 兼容层: 原始 ADMETPrediction 类 =====

class ADMETPrediction:
    """
    ADMET 性质预测兼容类
    
    此类为旧代码提供向后兼容性，实际功能委托给新的 ADMETPredictor
    """
    
    def __init__(self, admet_type='Solubility', model_path=None):
        """
        初始化 ADMET 预测器 (兼容版本)
        
        Args:
            admet_type: ADMET 性质类型 ('Solubility', 'MetabolicStability')
            model_path: 模型文件路径（可选）
        """
        warnings.warn(
            "ADMETPrediction 已弃用，请使用 pharmaai.predictors.admet.ADMETPredictor",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.admet_type = admet_type
        
        # 映射旧类型到新枚举
        admet_map = {
            'Solubility': ADMETType.SOLUBILITY,
            'MetabolicStability': ADMETType.METABOLIC_STABILITY
        }
        
        mapped_type = admet_map.get(admet_type, ADMETType.SOLUBILITY)
        
        # 创建新的预测器实例
        self._predictor = NewADMETPredictor(admet_type=mapped_type)
        
        # 加载模型文件（如果提供）
        if model_path:
            self._predictor.load_model(model_path)
    
    def train(self, df: pd.DataFrame, label_column=None, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """
        训练 ADMET 预测模型 (兼容方法)
        
        Args:
            df: 包含 'smiles' 和标签列的 DataFrame
            label_column: 标签列名
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        return self._predictor.train(df, label_column=label_column, test_size=test_size, random_state=random_state)
    
    def predict(self, smiles: str) -> Dict[str, Any]:
        """
        预测单个分子的 ADMET 性质 (兼容方法)
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            包含预测结果的字典
        """
        result = self._predictor.predict(smiles)
        
        # 转换为旧格式
        return {
            'smiles': smiles,
            'predicted_value': result.value,
            'confidence': result.confidence,
            'admet_type': self.admet_type,
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
                'predicted_value': result.value,
                'confidence': result.confidence,
                'admet_type': self.admet_type,
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
    
    def get_molecular_features(self, smiles: str) -> Dict[str, float]:
        """
        获取分子特征 (兼容方法)
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            特征字典
        """
        result = self._predictor.predict(smiles)
        features = result.metadata.get('feature_dict', {})
        return features


# ===== 兼容层: 原有全局函数 =====

def load_admet_model(admet_type='Solubility', model_path=None) -> ADMETPrediction:
    """
    加载 ADMET 模型 (兼容函数)
    
    Args:
        admet_type: ADMET 性质类型
        model_path: 模型文件路径
        
    Returns:
        ADMETPrediction 实例
    """
    warnings.warn(
        "load_admet_model 已弃用，请使用 pharmaai.predictors.admet.create_admet_predictor",
        DeprecationWarning,
        stacklevel=2
    )
    return ADMETPrediction(admet_type=admet_type, model_path=model_path)


def predict_admet(smiles: str, admet_type='Solubility', model_path=None) -> Dict[str, Any]:
    """
    预测 ADMET 性质 (兼容函数)
    
    Args:
        smiles: SMILES 字符串
        admet_type: ADMET 性质类型
        model_path: 模型文件路径
        
    Returns:
        包含预测结果的字典
    """
    warnings.warn(
        "predict_admet 已弃用，请使用 pharmaai.predictors.admet.ADMETPredictor",
        DeprecationWarning,
        stacklevel=2
    )
    
    predictor = ADMETPrediction(admet_type=admet_type, model_path=model_path)
    return predictor.predict(smiles)


def batch_predict_admet(smiles_list: List[str], admet_type='Solubility', model_path=None) -> List[Dict[str, Any]]:
    """
    批量预测 ADMET 性质 (兼容函数)
    
    Args:
        smiles_list: SMILES 字符串列表
        admet_type: ADMET 性质类型
        model_path: 模型文件路径
        
    Returns:
        预测结果列表
    """
    warnings.warn(
        "batch_predict_admet 已弃用，请使用 pharmaai.predictors.admet.ADMETPredictor.batch_predict",
        DeprecationWarning,
        stacklevel=2
    )
    
    predictor = ADMETPrediction(admet_type=admet_type, model_path=model_path)
    return predictor.batch_predict(smiles_list)


# ===== 保持原有模块属性 =====

__all__ = [
    'ADMETPrediction',
    'load_admet_model',
    'predict_admet',
    'batch_predict_admet',
    'ADMETPredictor',  # 新预测器类（如果导入成功）
    'ADMETType',       # 新枚举类型（如果导入成功）
    'create_admet_predictor'  # 新创建函数（如果导入成功）
]

# 如果新预测器导入成功，将其导出
try:
    ADMETPredictor = NewADMETPredictor
except NameError:
    pass

if __name__ == '__main__':
    # 兼容性测试
    print("ADMET 预测模块兼容层已加载")
    print("注意: 此模块已弃用，请迁移到 pharmaai.predictors.admet")