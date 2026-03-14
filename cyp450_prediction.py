#!/usr/bin/env python3
"""
CYP450 抑制预测模块 (兼容层)
已弃用 - 请使用新的 pharmaai.predictors.cyp450 模块

此模块已迁移到统一架构中，此文件仅为向后兼容而保留
新代码应导入: from pharmaai.predictors.cyp450 import CYP450Predictor
"""

import warnings
warnings.warn(
    "cyp450_prediction.py 已弃用，请使用 pharmaai.predictors.cyp450",
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
    from pharmaai.predictors.cyp450 import CYP450Predictor as NewCYP450Predictor
    from pharmaai.predictors.cyp450 import create_cyp450_predictor
except ImportError as e:
    warnings.warn(f"无法导入新模块: {e}", ImportWarning)
    # 定义兼容的占位符类，防止导入失败
    class NewCYP450Predictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("CYP450Predictor 需要安装新模块。请使用 pip install -e .")
    
    def create_cyp450_predictor(*args, **kwargs):
        raise ImportError("create_cyp450_predictor 需要安装新模块。请使用 pip install -e .")


# ===== 兼容层: 原始 CYP450Prediction 类 =====

class CYP450Prediction:
    """
    CYP450 抑制预测兼容类
    
    此类为旧代码提供向后兼容性，实际功能委托给新的 CYP450Predictor
    """
    
    def __init__(self, isoform='CYP3A4', model_path=None):
        """
        初始化 CYP450 预测器 (兼容版本)
        
        Args:
            isoform: CYP亚型 ('CYP3A4', 'CYP2D6', 'CYP2C9')
            model_path: 模型文件路径（可选）
        """
        warnings.warn(
            "CYP450Prediction 已弃用，请使用 pharmaai.predictors.cyp450.CYP450Predictor",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.isoform = isoform
        self.model_path = model_path
        
        # 创建新的预测器实例
        self._predictor = NewCYP450Predictor(isoform=isoform)
        
        # 加载模型文件（如果提供）
        if model_path:
            self._predictor.load_model(model_path)
    
    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """
        训练 CYP450 预测模型 (兼容方法)
        
        Args:
            df: 包含 'smiles' 和 'is_inhibitor' 列的 DataFrame
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        return self._predictor.train(df, test_size=test_size, random_state=random_state)
    
    def predict(self, smiles: str) -> Dict[str, Any]:
        """
        预测单个分子的 CYP450 抑制作用 (兼容方法)
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            包含预测结果的字典
        """
        result = self._predictor.predict(smiles)
        
        # 转换为旧格式
        return {
            'smiles': smiles,
            'is_inhibitor': result.value,
            'confidence': result.confidence,
            'isoform': self.isoform,
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
                'is_inhibitor': result.value,
                'confidence': result.confidence,
                'isoform': self.isoform,
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
        self.model_path = filepath
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


# ===== 兼容层: 原有全局函数 =====

def load_cyp450_model(isoform='CYP3A4', model_path=None) -> CYP450Prediction:
    """
    加载 CYP450 模型 (兼容函数)
    
    Args:
        isoform: CYP亚型
        model_path: 模型文件路径
        
    Returns:
        CYP450Prediction 实例
    """
    warnings.warn(
        "load_cyp450_model 已弃用，请使用 pharmaai.predictors.cyp450.create_cyp450_predictor",
        DeprecationWarning,
        stacklevel=2
    )
    return CYP450Prediction(isoform=isoform, model_path=model_path)


def predict_cyp450_inhibition(smiles: str, isoform='CYP3A4', model_path=None) -> Dict[str, Any]:
    """
    预测 CYP450 抑制作用 (兼容函数)
    
    Args:
        smiles: SMILES 字符串
        isoform: CYP亚型
        model_path: 模型文件路径
        
    Returns:
        包含预测结果的字典
    """
    warnings.warn(
        "predict_cyp450_inhibition 已弃用，请使用 pharmaai.predictors.cyp450.CYP450Predictor",
        DeprecationWarning,
        stacklevel=2
    )
    
    predictor = CYP450Prediction(isoform=isoform, model_path=model_path)
    return predictor.predict(smiles)


# ===== 保持原有模块属性 =====

__all__ = [
    'CYP450Prediction',
    'load_cyp450_model',
    'predict_cyp450_inhibition',
    'CYP450Predictor',  # 新预测器类（如果导入成功）
    'create_cyp450_predictor'  # 新创建函数（如果导入成功）
]

# 如果新预测器导入成功，将其导出
try:
    CYP450Predictor = NewCYP450Predictor
except NameError:
    pass

if __name__ == '__main__':
    # 兼容性测试
    print("CYP450 预测模块兼容层已加载")
    print("注意: 此模块已弃用，请迁移到 pharmaai.predictors.cyp450")