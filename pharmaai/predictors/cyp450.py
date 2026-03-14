"""
CYP450 预测器模块

基于统一 core 架构的 CYP450 抑制预测器
继承自 BasePredictor，使用 MorganFingerprintGenerator 单例和 Settings 配置管理
"""

import warnings
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rdkit import Chem

from pharmaai.core.base_predictor import SklearnPredictor, PredictionResult
from pharmaai.core.utils import MorganFingerprintGenerator, calculate_molecular_features
from pharmaai.core.config import Settings

logger = logging.getLogger(__name__)


class CYP450Predictor(SklearnPredictor):
    """
    CYP450 抑制预测器
    
    支持多种 CYP 亚型 (CYP3A4, CYP2D6, CYP2C9) 的抑制预测
    使用 Morgan 指纹和分子描述符作为特征，随机森林分类器
    
    Example:
        >>> predictor = CYP450Predictor(isoform='CYP3A4')
        >>> predictor.train(training_data)
        >>> result = predictor.predict("CCO")
        >>> print(result.value, result.confidence)
    """
    
    def __init__(self, isoform: str = 'CYP3A4', model_name: str = None, version: str = "1.0.0"):
        """
        初始化 CYP450 预测器
        
        Args:
            isoform: CYP亚型 ('CYP3A4', 'CYP2D6', 'CYP2C9')
            model_name: 模型名称，默认为 'cyp450_{isoform}'
            version: 模型版本
        """
        if isoform not in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            raise ValueError(f"Unsupported CYP isoform: {isoform}")
        
        self.isoform = isoform
        self.settings = Settings()
        model_name = model_name or f"cyp450_{isoform.lower()}"
        super().__init__(model_name=model_name, version=version)
        
        # 初始化 Morgan 指纹生成器（单例）
        self.fp_generator = MorganFingerprintGenerator.get_instance(
            radius=self.settings.fingerprint_radius,
            fp_size=self.settings.fingerprint_size
        )
        
        # 特征列（训练后设置）
        self.desc_cols = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
            'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
            'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]
        
        # 阈值
        self.threshold = self.settings.cyp450_threshold
        
        logger.debug(f"CYP450Predictor initialized for {isoform}")
    
    def _extract_features(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """
        提取分子特征（实现 SklearnPredictor 抽象方法）
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            合并的描述符和指纹特征数组
        """
        # 计算分子描述符
        features = calculate_molecular_features(mol, include_fingerprint=False)
        if features is None:
            raise ValueError("无法计算分子特征")
        
        # 提取描述符值
        desc_values = [features.get(col, 0.0) for col in self.desc_cols]
        
        # 计算 Morgan 指纹
        fp = self.fp_generator.generate(mol)
        if fp is None:
            raise ValueError("无法生成分子指纹")
        
        # 合并特征
        return np.concatenate([desc_values, fp])
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        训练 CYP450 预测模型
        
        Args:
            df: 包含 'smiles' 和 'is_inhibitor' 列的 DataFrame
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        logger.info(f"开始训练 {self.isoform} 预测模型...")
        
        # 准备特征矩阵和标签
        X_list = []
        y_list = []
        smiles_list = []
        
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                logger.warning(f"无效的 SMILES: {row['smiles']}")
                continue
            
            try:
                features = self._extract_features(mol)
                X_list.append(features)
                y_list.append(row['is_inhibitor'])
                smiles_list.append(row['smiles'])
            except Exception as e:
                logger.warning(f"特征提取失败: {row['smiles']}, 错误: {e}")
                continue
        
        if not X_list:
            return {'error': '无法从数据中提取有效特征'}
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征维度: {X.shape[1]}")
        logger.info(f"  抑制剂比例: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 训练随机森林模型
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'
        )
        
        self._model.fit(X_train, y_train)
        self._is_trained = True
        
        # 评估模型
        train_score = self._model.score(X_train, y_train)
        test_score = self._model.score(X_test, y_test)
        
        # 特征重要性
        self.feature_importance = dict(zip(
            self.desc_cols + [f'morgan_{i}' for i in range(self.settings.fingerprint_size)],
            self._model.feature_importances_
        ))
        
        # 预测结果
        y_pred = self._model.predict(X_test)
        y_pred_proba = self._model.predict_proba(X_test)[:, 1]
        
        result = {
            'model_name': self.model_name,
            'isoform': self.isoform,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'feature_importance': self.feature_importance,
            'threshold': self.threshold,
            'is_trained': True
        }
        
        logger.info(f"训练完成: 训练集准确率 {train_score:.3f}, 测试集准确率 {test_score:.3f}")
        return result
    
    def predict(self, mol: Union[str, Chem.rdchem.Mol]) -> PredictionResult:
        """
        对单个分子进行 CYP450 抑制预测
        
        Args:
            mol: SMILES 字符串或 RDKit 分子对象
            
        Returns:
            PredictionResult: 预测结果
        """
        self._check_trained()
        
        mol_obj = self.validate_molecule(mol)
        if mol_obj is None:
            return PredictionResult(
                value=False,
                confidence=0.0,
                model_name=self.model_name,
                metadata={'error': 'Invalid molecule', 'isoform': self.isoform}
            )
        
        features = self._extract_features(mol_obj)
        
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba([features])[0]
            pred_class = proba[1] >= self.threshold
            confidence = proba[1] if pred_class else proba[0]
        else:
            pred_class = self._model.predict([features])[0]
            confidence = None
        
        metadata = {
            'isoform': self.isoform,
            'threshold': self.threshold,
            'feature_count': len(features)
        }
        
        return PredictionResult(
            value=bool(pred_class),
            confidence=float(confidence) if confidence is not None else None,
            model_name=self.model_name,
            metadata=metadata
        )
    
    def batch_predict(
        self, 
        molecules: List[Union[str, Chem.rdchem.Mol]]
    ) -> List[PredictionResult]:
        """
        批量预测（覆盖基类方法以优化性能）
        
        Args:
            molecules: 分子列表
            
        Returns:
            预测结果列表
        """
        self._check_trained()
        
        results = []
        for mol in molecules:
            try:
                results.append(self.predict(mol))
            except Exception as e:
                logger.warning(f"批量预测失败: {mol}, 错误: {e}")
                results.append(PredictionResult(
                    value=None,
                    confidence=0.0,
                    model_name=self.model_name,
                    metadata={'error': str(e), 'isoform': self.isoform}
                ))
        
        return results
    
    def save_model(self, filepath: Union[str, Path]) -> bool:
        """
        保存 CYP450 预测模型
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        import joblib
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self._model,
                'model_name': self.model_name,
                'isoform': self.isoform,
                'version': self.version,
                'is_trained': self._is_trained,
                'desc_cols': self.desc_cols,
                'threshold': self.threshold,
                'settings': {
                    'fingerprint_radius': self.settings.fingerprint_radius,
                    'fingerprint_size': self.settings.fingerprint_size
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"模型保存到 {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: Union[str, Path]) -> bool:
        """
        加载 CYP450 预测模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            是否加载成功
        """
        import joblib
        
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self._model = model_data['model']
            self.model_name = model_data.get('model_name', self.model_name)
            self.isoform = model_data.get('isoform', self.isoform)
            self._is_trained = model_data.get('is_trained', False)
            self.desc_cols = model_data.get('desc_cols', self.desc_cols)
            self.threshold = model_data.get('threshold', self.threshold)
            
            logger.info(f"模型从 {filepath} 加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        return self.desc_cols + [f'morgan_{i}' for i in range(self.settings.fingerprint_size)]


def create_cyp450_predictor(isoform: str = 'CYP3A4') -> CYP450Predictor:
    """
    创建 CYP450 预测器的便捷函数
    
    Args:
        isoform: CYP亚型
        
    Returns:
        CYP450Predictor 实例
    """
    return CYP450Predictor(isoform=isoform)