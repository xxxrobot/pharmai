"""
PharmaAI 基础预测器抽象类

提供统一的预测器接口，所有具体预测器都应该继承此类
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    预测结果数据类
    
    Attributes:
        value: 预测值（分类概率或回归值）
        confidence: 置信度（0-1之间）
        model_name: 使用的模型名称
        metadata: 额外元数据
    """
    value: Union[float, int, bool, str]
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证置信度范围"""
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            logger.warning(f"Confidence value {self.confidence} is outside [0, 1] range")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'value': self.value,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return f"PredictionResult(value={self.value}, confidence={self.confidence:.3f})"


class BasePredictor(ABC):
    """
    基础预测器抽象类
    
    所有预测器（毒性预测、ADMET预测等）都应该继承此类
    并实现以下抽象方法：
    - predict: 单分子预测
    - batch_predict: 批量预测
    - save_model: 保存模型
    - load_model: 加载模型
    
    Example:
        >>> class ToxicityPredictor(BasePredictor):
        ...     def predict(self, mol):
        ...         # 实现预测逻辑
        ...         return PredictionResult(value=0.8, confidence=0.95)
    """
    
    def __init__(self, model_name: str = "base_predictor", version: str = "1.0.0"):
        """
        初始化预测器
        
        Args:
            model_name: 模型名称
            version: 模型版本
        """
        self.model_name = model_name
        self.version = version
        self._is_trained = False
        self._model = None
        
        logger.debug(f"Initialized {model_name} predictor (v{version})")
    
    @property
    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self._is_trained
    
    @abstractmethod
    def predict(self, mol: Union[str, Chem.rdchem.Mol]) -> PredictionResult:
        """
        对单个分子进行预测
        
        Args:
            mol: SMILES字符串或RDKit分子对象
            
        Returns:
            PredictionResult: 预测结果
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def batch_predict(
        self, 
        molecules: List[Union[str, Chem.rdchem.Mol]]
    ) -> List[PredictionResult]:
        """
        对多个分子进行批量预测
        
        Args:
            molecules: SMILES字符串或RDKit分子对象列表
            
        Returns:
            List[PredictionResult]: 预测结果列表
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: Union[str, Path]) -> bool:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            bool: 是否保存成功
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: Union[str, Path]) -> bool:
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            bool: 是否加载成功
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    def validate_molecule(self, mol: Union[str, Chem.rdchem.Mol]) -> Optional[Chem.rdchem.Mol]:
        """
        验证并转换分子输入
        
        Args:
            mol: SMILES字符串或RDKit分子对象
            
        Returns:
            RDKit分子对象或None（如果无效）
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            logger.warning("Invalid molecule provided")
            return None
        
        return mol
    
    def _check_trained(self) -> None:
        """
        检查模型是否已训练
        
        Raises:
            RuntimeError: 如果模型未训练
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Model '{self.model_name}' is not trained. "
                "Please train the model or load a pre-trained model first."
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self._is_trained,
            'model_type': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', trained={self._is_trained})"


class SklearnPredictor(BasePredictor):
    """
    基于scikit-learn的预测器基类
    
    为使用sklearn模型的预测器提供通用实现
    """
    
    def __init__(self, model_name: str = "sklearn_predictor", version: str = "1.0.0"):
        super().__init__(model_name, version)
        self._threshold = 0.5
    
    def predict(self, mol: Union[str, Chem.rdchem.Mol]) -> PredictionResult:
        """
        使用sklearn模型进行预测
        
        Args:
            mol: SMILES字符串或RDKit分子对象
            
        Returns:
            PredictionResult: 预测结果
        """
        self._check_trained()
        
        mol_obj = self.validate_molecule(mol)
        if mol_obj is None:
            return PredictionResult(
                value=None,
                confidence=0.0,
                model_name=self.model_name,
                metadata={'error': 'Invalid molecule'}
            )
        
        # 子类应该实现特征提取
        features = self._extract_features(mol_obj)
        
        # 预测
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba([features])[0]
            pred_class = self._model.predict([features])[0]
            confidence = max(proba)
        else:
            pred_class = self._model.predict([features])[0]
            confidence = None
        
        return PredictionResult(
            value=pred_class,
            confidence=confidence,
            model_name=self.model_name,
            metadata={'features_shape': features.shape if hasattr(features, 'shape') else len(features)}
        )
    
    def batch_predict(
        self, 
        molecules: List[Union[str, Chem.rdchem.Mol]]
    ) -> List[PredictionResult]:
        """
        批量预测
        
        Args:
            molecules: 分子列表
            
        Returns:
            预测结果列表
        """
        return [self.predict(mol) for mol in molecules]
    
    def save_model(self, filepath: Union[str, Path]) -> bool:
        """
        使用pickle保存sklearn模型
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self._model,
                'model_name': self.model_name,
                'version': self.version,
                'is_trained': self._is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: Union[str, Path]) -> bool:
        """
        使用pickle加载sklearn模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            filepath = Path(filepath)
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self._model = model_data['model']
            self.model_name = model_data.get('model_name', self.model_name)
            self.version = model_data.get('version', self.version)
            self._is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @abstractmethod
    def _extract_features(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """
        提取分子特征（子类必须实现）
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            特征数组
        """
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Set self._model first.")
        
        self._model.fit(X, y)
        self._is_trained = True
        logger.info(f"Model {self.model_name} trained on {len(X)} samples")


__all__ = [
    "BasePredictor",
    "SklearnPredictor", 
    "PredictionResult"
]
