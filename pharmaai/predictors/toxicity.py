"""
毒性预测器模块

基于统一 core 架构的毒性预测器
继承自 BasePredictor，支持多个毒性终点：
- hERG 心脏毒性
- 肝毒性 (Hepatotoxicity)
- 致突变性 (Ames)

使用 MorganFingerprintGenerator 单例、Settings 配置管理和统一的特征准备函数
"""

import warnings
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rdkit import Chem

from pharmaai.core.base_predictor import SklearnPredictor, PredictionResult
from pharmaai.core.utils import MorganFingerprintGenerator, calculate_molecular_features
from pharmaai.core.config import Settings

logger = logging.getLogger(__name__)


class ToxicityType(Enum):
    """毒性类型枚举"""
    HERG = "hERG"
    HEPATOTOXICITY = "Hepatotoxicity"
    AMES = "Ames"


class ToxicityPredictor(SklearnPredictor):
    """
    多终点毒性预测器
    
    支持三种毒性终点：
    - hERG 心脏毒性预测
    - 肝毒性 (Hepatotoxicity) 预测
    - 致突变性 (Ames) 预测
    
    每个终点使用独立的随机森林模型，共享特征提取逻辑
    
    Example:
        >>> predictor = ToxicityPredictor(toxicity_type=ToxicityType.HERG)
        >>> predictor.train(training_data, toxicity_type=ToxicityType.HERG)
        >>> result = predictor.predict("CCO")
        >>> print(result.value, result.confidence)
    """
    
    # 毒性相关警示结构 SMARTS 模式
    TOXICITY_SMARTS = {
        'basic_amine': '[NX3;H2,H1;!$(NC=O)]',
        'aromatic_amine': '[NX3;H2,H1]c1ccccc1',
        'piperazine': 'C1CNCCN1',
        'quaternary_n': '[N+;X4]',
        'nitro': '[N+](=O)[O-]',
        'halogenated_aromatic': 'c[F,Cl,Br,I]',
        'hydrazine': '[NX3H2][NX3H2]',
        'alkyl_halide': 'C[F,Cl,Br,I]',
        'epoxide': 'C1OC1',
        'azo': '[N+]=[N-]',
    }
    
    # 毒性相关特征列
    TOXICITY_DESC_COLS = [
        # 物理化学性质
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
        'NumHalogens', 'NumSulfurs', 'NumOxygens',
        
        # 毒性警示结构
        'has_basic_amine', 'has_aromatic_amine', 'has_piperazine',
        'has_quaternary_n', 'has_nitro', 'has_halogenated_aromatic',
        'has_hydrazine', 'has_alkyl_halide', 'has_epoxide', 'has_azo'
    ]
    
    def __init__(self, toxicity_type: ToxicityType = ToxicityType.HERG,
                 model_name: str = None, version: str = "1.0.0"):
        """
        初始化毒性预测器
        
        Args:
            toxicity_type: 毒性类型 (hERG, Hepatotoxicity, Ames)
            model_name: 模型名称，默认为 'toxicity_{type}'
            version: 模型版本
        """
        self.toxicity_type = toxicity_type
        self.settings = Settings()
        model_name = model_name or f"toxicity_{toxicity_type.value.lower()}"
        super().__init__(model_name=model_name, version=version)
        
        # 初始化 Morgan 指纹生成器（单例）
        self.fp_generator = MorganFingerprintGenerator.get_instance(
            radius=self.settings.fingerprint_radius,
            fp_size=self.settings.fingerprint_size
        )
        
        # 初始化警示结构模式
        self._init_smarts_patterns()
        
        # 阈值
        self.threshold = self.settings.toxicity_threshold
        
        logger.debug(f"ToxicityPredictor initialized for {toxicity_type.value}")
    
    def _init_smarts_patterns(self):
        """初始化 SMARTS 模式"""
        self._smarts_patterns = {}
        for name, smarts in self.TOXICITY_SMARTS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                self._smarts_patterns[name] = pattern
            except Exception as e:
                logger.warning(f"SMARTS 模式解析失败 {name}: {smarts}, 错误: {e}")
                self._smarts_patterns[name] = None
    
    def _check_smarts_match(self, mol: Chem.rdchem.Mol, pattern_name: str) -> bool:
        """
        检查分子是否匹配特定的 SMARTS 模式
        
        Args:
            mol: RDKit 分子对象
            pattern_name: 模式名称
            
        Returns:
            是否匹配
        """
        pattern = self._smarts_patterns.get(pattern_name)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
        return False
    
    def _extract_features(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """
        提取毒性相关分子特征（实现 SklearnPredictor 抽象方法）
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            合并的描述符、指纹和警示结构特征数组
        """
        # 计算基础分子描述符
        features = calculate_molecular_features(mol, include_fingerprint=False)
        if features is None:
            raise ValueError("无法计算分子特征")
        
        # 提取基础描述符值
        base_desc = [features.get(col, 0.0) for col in [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
            'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
            'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]]
        
        # 计算毒性警示结构特征
        alert_features = []
        for alert_name in [
            'basic_amine', 'aromatic_amine', 'piperazine',
            'quaternary_n', 'nitro', 'halogenated_aromatic',
            'hydrazine', 'alkyl_halide', 'epoxide', 'azo'
        ]:
            alert_features.append(float(self._check_smarts_match(mol, alert_name)))
        
        # 计算 Morgan 指纹
        fp = self.fp_generator.generate(mol)
        if fp is None:
            raise ValueError("无法生成分子指纹")
        
        # 合并所有特征
        return np.concatenate([base_desc, alert_features, fp])
    
    def _prepare_training_data(
        self, df: pd.DataFrame, toxicity_type: ToxicityType
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            df: 包含 'smiles' 和毒性标签列的 DataFrame
            toxicity_type: 毒性类型
            
        Returns:
            (特征矩阵, 标签数组, 有效SMILES列表)
        """
        # 确定标签列名
        label_column = f"is_{toxicity_type.value.lower()}"
        if label_column not in df.columns:
            raise ValueError(f"DataFrame 缺少标签列: {label_column}")
        
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
                y_list.append(row[label_column])
                smiles_list.append(row['smiles'])
            except Exception as e:
                logger.warning(f"特征提取失败: {row['smiles']}, 错误: {e}")
                continue
        
        if not X_list:
            raise ValueError("无法从数据中提取有效特征")
        
        return np.array(X_list), np.array(y_list), smiles_list
    
    def train(
        self, 
        df: pd.DataFrame, 
        toxicity_type: Optional[ToxicityType] = None,
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        训练毒性预测模型
        
        Args:
            df: 包含 'smiles' 和毒性标签列的 DataFrame
            toxicity_type: 毒性类型（如果为None，则使用初始化时指定的类型）
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        if toxicity_type is None:
            toxicity_type = self.toxicity_type
        
        logger.info(f"开始训练 {toxicity_type.value} 毒性预测模型...")
        
        # 准备数据
        X, y, smiles_list = self._prepare_training_data(df, toxicity_type)
        
        # 更新毒性类型（如果与初始化时不同）
        if toxicity_type != self.toxicity_type:
            self.toxicity_type = toxicity_type
            self.model_name = f"toxicity_{toxicity_type.value.lower()}"
        
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征维度: {X.shape[1]}")
        logger.info(f"  阳性样本比例: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
        
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
            self.get_feature_names(),
            self._model.feature_importances_
        ))
        
        result = {
            'model_name': self.model_name,
            'toxicity_type': toxicity_type.value,
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
        对单个分子进行毒性预测
        
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
                metadata={'error': 'Invalid molecule', 'toxicity_type': self.toxicity_type.value}
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
            'toxicity_type': self.toxicity_type.value,
            'threshold': self.threshold,
            'feature_count': len(features),
            'alert_features': {
                alert_name: bool(self._check_smarts_match(mol_obj, alert_name))
                for alert_name in self.TOXICITY_SMARTS.keys()
            }
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
                    metadata={'error': str(e), 'toxicity_type': self.toxicity_type.value}
                ))
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        base_names = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
            'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
            'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]
        
        alert_names = [
            'has_basic_amine', 'has_aromatic_amine', 'has_piperazine',
            'has_quaternary_n', 'has_nitro', 'has_halogenated_aromatic',
            'has_hydrazine', 'has_alkyl_halide', 'has_epoxide', 'has_azo'
        ]
        
        fp_names = [f'morgan_{i}' for i in range(self.settings.fingerprint_size)]
        
        return base_names + alert_names + fp_names
    
    def save_model(self, filepath: Union[str, Path]) -> bool:
        """
        保存毒性预测模型
        
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
                'toxicity_type': self.toxicity_type.value,
                'version': self.version,
                'is_trained': self._is_trained,
                'threshold': self.threshold,
                'smarts_patterns': self.TOXICITY_SMARTS,
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
        加载毒性预测模型
        
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
            self.toxicity_type = ToxicityType(model_data.get('toxicity_type', self.toxicity_type.value))
            self._is_trained = model_data.get('is_trained', False)
            self.threshold = model_data.get('threshold', self.threshold)
            
            logger.info(f"模型从 {filepath} 加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False


def create_toxicity_predictor(toxicity_type: ToxicityType = ToxicityType.HERG) -> ToxicityPredictor:
    """
    创建毒性预测器的便捷函数
    
    Args:
        toxicity_type: 毒性类型
        
    Returns:
        ToxicityPredictor 实例
    """
    return ToxicityPredictor(toxicity_type=toxicity_type)