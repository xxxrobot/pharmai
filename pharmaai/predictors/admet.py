"""
ADMET 预测器模块

基于统一 core 架构的 ADMET 性质预测器
继承自 BasePredictor，支持多种 ADMET 性质预测：
- 溶解度 (Solubility)
- 代谢稳定性 (Metabolic Stability)

使用 MorganFingerprintGenerator 单例和 Settings 配置管理
"""

import warnings
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from rdkit import Chem

from pharmaai.core.base_predictor import BasePredictor, PredictionResult
from pharmaai.core.utils import MorganFingerprintGenerator, calculate_molecular_features
from pharmaai.core.config import Settings

logger = logging.getLogger(__name__)


class ADMETType(Enum):
    """ADMET 性质类型枚举"""
    SOLUBILITY = "Solubility"
    METABOLIC_STABILITY = "MetabolicStability"
    # 其他 ADMET 性质可以在此扩展


class ADMETPredictor(BasePredictor):
    """
    ADMET 性质预测器
    
    支持多种 ADMET 性质预测：
    - 溶解度 (LogS, 水溶性) - 回归任务
    - 代谢稳定性 (清除率或半衰期) - 回归/分类任务
    
    使用统一特征提取和模型管理架构
    
    Example:
        >>> predictor = ADMETPredictor(admet_type=ADMETType.SOLUBILITY)
        >>> predictor.train(training_data)
        >>> result = predictor.predict("CCO")
        >>> print(result.value, result.confidence)
    """
    
    # ADMET 相关特征列
    ADMET_DESC_COLS = [
        # 物理化学性质
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
        'NumHalogens', 'NumSulfurs', 'NumOxygens',
        
        # 溶解度特定特征
        'TPSA_ratio', 'RotBonds_ratio', 'Aromatic_ratio',
        'MolRefractivity', 'HydrophilicFactor',
        'LogS_estimated', 'Polarity_index',
        
        # 代谢稳定性特定特征
        'CYP450_affinity_score', 'Metabolic_sites',
        'Hydroxylation_potential', 'Glucuronidation_potential'
    ]
    
    def __init__(self, admet_type: ADMETType = ADMETType.SOLUBILITY,
                 model_name: str = None, version: str = "1.0.0"):
        """
        初始化 ADMET 预测器
        
        Args:
            admet_type: ADMET 性质类型
            model_name: 模型名称，默认为 'admet_{type}'
            version: 模型版本
        """
        self.admet_type = admet_type
        self.settings = Settings()
        model_name = model_name or f"admet_{admet_type.value.lower()}"
        super().__init__(model_name=model_name, version=version)
        
        # 初始化 Morgan 指纹生成器（单例）
        self.fp_generator = MorganFingerprintGenerator.get_instance(
            radius=self.settings.fingerprint_radius,
            fp_size=self.settings.fingerprint_size
        )
        
        # 根据任务类型初始化模型
        if admet_type == ADMETType.SOLUBILITY:
            # 溶解度预测：回归任务
            self.task_type = 'regression'
            self._model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.threshold = None  # 回归任务无阈值
        elif admet_type == ADMETType.METABOLIC_STABILITY:
            # 代谢稳定性预测：分类任务（稳定/不稳定）
            self.task_type = 'classification'
            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            self.threshold = 0.5  # 分类阈值
        else:
            raise ValueError(f"不支持的 ADMET 类型: {admet_type}")
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self._is_fitted_scaler = False
        
        # 特征列（实际使用的列）
        self.feature_cols = []
        
        logger.debug(f"ADMETPredictor initialized for {admet_type.value}")
    
    def _extract_admet_features(self, mol: Chem.rdchem.Mol) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        提取 ADMET 相关分子特征
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            (特征数组, 特征字典)
        """
        # 计算基础分子描述符
        features = calculate_molecular_features(mol, include_fingerprint=False)
        if features is None:
            raise ValueError("无法计算分子特征")
        
        # 计算 ADMET 特定特征
        admet_features = self._calculate_admet_specific_features(mol, features)
        
        # 计算 Morgan 指纹
        fp = self.fp_generator.generate(mol)
        if fp is None:
            raise ValueError("无法生成分子指纹")
        
        # 合并所有特征值
        all_features = []
        feature_dict = {}
        
        # 基础描述符
        for col in [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
            'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
            'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]:
            value = features.get(col, 0.0)
            all_features.append(value)
            feature_dict[col] = value
        
        # ADMET 特定特征
        for col in [
            'TPSA_ratio', 'RotBonds_ratio', 'Aromatic_ratio',
            'MolRefractivity', 'HydrophilicFactor',
            'LogS_estimated', 'Polarity_index',
            'CYP450_affinity_score', 'Metabolic_sites',
            'Hydroxylation_potential', 'Glucuronidation_potential'
        ]:
            value = admet_features.get(col, 0.0)
            all_features.append(value)
            feature_dict[col] = value
        
        # 指纹特征
        for i, fp_value in enumerate(fp):
            all_features.append(fp_value)
            feature_dict[f'morgan_{i}'] = fp_value
        
        return np.array(all_features), feature_dict
    
    def _calculate_admet_specific_features(
        self, mol: Chem.rdchem.Mol, base_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算 ADMET 特定特征
        
        Args:
            mol: RDKit 分子对象
            base_features: 基础特征字典
            
        Returns:
            ADMET 特定特征字典
        """
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        import math
        
        admet_features = {}
        
        # 溶解度相关特征
        mw = base_features.get('MW', 0.0)
        logp = base_features.get('LogP', 0.0)
        tpsa = base_features.get('TPSA', 0.0)
        hbd = base_features.get('HBD', 0.0)
        hba = base_features.get('HBA', 0.0)
        rot_bonds = base_features.get('RotatableBonds', 0.0)
        aromatic_rings = base_features.get('AromaticRings', 0.0)
        ring_count = Lipinski.RingCount(mol) if mol else 0
        
        # TPSA 比例
        admet_features['TPSA_ratio'] = tpsa / mw if mw > 0 else 0.0
        
        # 可旋转键比例
        admet_features['RotBonds_ratio'] = rot_bonds / mw * 100 if mw > 0 else 0.0
        
        # 芳香环比例
        admet_features['Aromatic_ratio'] = aromatic_rings / ring_count if ring_count > 0 else 0.0
        
        # 分子折射率
        try:
            admet_features['MolRefractivity'] = Crippen.MolMR(mol)
        except:
            admet_features['MolRefractivity'] = 0.0
        
        # 亲水性因子
        admet_features['HydrophilicFactor'] = tpsa / (logp + 1) if logp > -1 else tpsa
        
        # 估算 LogS
        admet_features['LogS_estimated'] = 0.5 - 0.01 * (mw - 100) - 0.5 * logp
        
        # 极性指数
        admet_features['Polarity_index'] = tpsa / (hbd + hba + 1)
        
        # 代谢稳定性相关特征
        # CYP450 亲和力评分（基于分子属性估算）
        cyp450_score = 0.0
        if logp > 2.0:
            cyp450_score += 0.3
        if aromatic_rings > 1:
            cyp450_score += 0.2
        if mol:
            nitrogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            if nitrogen_count > 0:
                cyp450_score += 0.2 * nitrogen_count
        admet_features['CYP450_affinity_score'] = min(cyp450_score, 1.0)
        
        # 代谢位点计数
        metabolic_sites = 0
        if mol:
            # 简单估算可代谢的碳原子
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # 碳原子
                    # 检查是否为叔碳或苄基碳
                    neighbors = atom.GetNeighbors()
                    h_count = sum(1 for n in neighbors if n.GetAtomicNum() == 1)
                    if h_count >= 1:
                        metabolic_sites += 1
        admet_features['Metabolic_sites'] = float(metabolic_sites)
        
        # 羟基化潜力
        admet_features['Hydroxylation_potential'] = 0.1 * hbd + 0.05 * hba
        
        # 葡萄糖醛酸化潜力
        admet_features['Glucuronidation_potential'] = 0.15 * hba
        
        return admet_features
    
    def _prepare_training_data(
        self, df: pd.DataFrame, label_column: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            df: 包含 'smiles' 和标签列的 DataFrame
            label_column: 标签列名
            
        Returns:
            (特征矩阵, 标签数组, 有效SMILES列表)
        """
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
                features, _ = self._extract_admet_features(mol)
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
        label_column: Optional[str] = None,
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        训练 ADMET 预测模型
        
        Args:
            df: 包含 'smiles' 和标签列的 DataFrame
            label_column: 标签列名（如果为None，根据ADMET类型自动选择）
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练结果的字典
        """
        # 确定标签列
        if label_column is None:
            if self.admet_type == ADMETType.SOLUBILITY:
                label_column = 'LogS'
            elif self.admet_type == ADMETType.METABOLIC_STABILITY:
                label_column = 'is_stable'
            else:
                raise ValueError(f"未知的 ADMET 类型: {self.admet_type}")
        
        logger.info(f"开始训练 {self.admet_type.value} 预测模型...")
        
        # 准备数据
        X, y, smiles_list = self._prepare_training_data(df, label_column)
        
        # 确定特征列（第一次训练时设置）
        if not self.feature_cols:
            self.feature_cols = self._get_feature_names_from_features(X.shape[1])
        
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征维度: {X.shape[1]}")
        logger.info(f"  任务类型: {self.task_type}")
        
        # 特征缩放
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self._is_fitted_scaler = True
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state,
            stratify=(y if self.task_type == 'classification' else None)
        )
        
        # 训练模型
        self._model.fit(X_train, y_train)
        self._is_trained = True
        
        # 评估模型
        if self.task_type == 'regression':
            # 回归任务评估
            y_train_pred = self._model.predict(X_train)
            y_test_pred = self._model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            metrics = {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2)
            }
        else:
            # 分类任务评估
            y_train_pred = self._model.predict(X_train)
            y_test_pred = self._model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            metrics = {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy)
            }
        
        # 特征重要性
        if hasattr(self._model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_cols,
                self._model.feature_importances_
            ))
        else:
            self.feature_importance = {}
        
        result = {
            'model_name': self.model_name,
            'admet_type': self.admet_type.value,
            'task_type': self.task_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'label_column': label_column,
            'feature_count': X.shape[1],
            **metrics,
            'feature_importance': self.feature_importance,
            'is_trained': True
        }
        
        logger.info(f"训练完成: {self._format_metrics(metrics)}")
        return result
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化评估指标"""
        if self.task_type == 'regression':
            return f"RMSE: 训练集 {metrics['train_rmse']:.3f}, 测试集 {metrics['test_rmse']:.3f}, R²: 训练集 {metrics['train_r2']:.3f}, 测试集 {metrics['test_r2']:.3f}"
        else:
            return f"准确率: 训练集 {metrics['train_accuracy']:.3f}, 测试集 {metrics['test_accuracy']:.3f}"
    
    def predict(self, mol: Union[str, Chem.rdchem.Mol]) -> PredictionResult:
        """
        对单个分子进行 ADMET 性质预测
        
        Args:
            mol: SMILES 字符串或 RDKit 分子对象
            
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
                metadata={'error': 'Invalid molecule', 'admet_type': self.admet_type.value}
            )
        
        # 提取特征
        features, feature_dict = self._extract_admet_features(mol_obj)
        
        # 特征缩放
        if self._is_fitted_scaler:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]
        
        # 预测
        if self.task_type == 'regression':
            # 回归任务
            pred_value = self._model.predict(features_scaled)[0]
            confidence = None
        else:
            # 分类任务
            if hasattr(self._model, 'predict_proba'):
                proba = self._model.predict_proba(features_scaled)[0]
                pred_class = proba[1] >= self.threshold
                pred_value = bool(pred_class)
                confidence = proba[1] if pred_class else proba[0]
            else:
                pred_class = self._model.predict(features_scaled)[0]
                pred_value = bool(pred_class)
                confidence = None
        
        metadata = {
            'admet_type': self.admet_type.value,
            'task_type': self.task_type,
            'feature_count': len(features),
            'feature_dict': feature_dict
        }
        
        return PredictionResult(
            value=pred_value,
            confidence=float(confidence) if confidence is not None else None,
            model_name=self.model_name,
            metadata=metadata
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
                    metadata={'error': str(e), 'admet_type': self.admet_type.value}
                ))
        
        return results
    
    def _get_feature_names_from_features(self, n_features: int) -> List[str]:
        """
        根据特征数量生成特征名称
        
        Args:
            n_features: 特征数量
            
        Returns:
            特征名称列表
        """
        feature_names = []
        
        # 基础描述符
        base_cols = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
            'AromaticRings', 'HeavyAtoms', 'NumNitrogens',
            'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]
        feature_names.extend(base_cols[:min(len(base_cols), 12)])
        
        # ADMET 特定特征
        admet_cols = [
            'TPSA_ratio', 'RotBonds_ratio', 'Aromatic_ratio',
            'MolRefractivity', 'HydrophilicFactor',
            'LogS_estimated', 'Polarity_index',
            'CYP450_affinity_score', 'Metabolic_sites',
            'Hydroxylation_potential', 'Glucuronidation_potential'
        ]
        feature_names.extend(admet_cols[:min(len(admet_cols), n_features - len(feature_names))])
        
        # 指纹特征
        fp_start_idx = len(feature_names)
        for i in range(n_features - fp_start_idx):
            feature_names.append(f'morgan_{i}')
        
        return feature_names
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        if self.feature_cols:
            return self.feature_cols
        else:
            # 如果没有训练，返回默认特征名称
            default_features = (
                self.ADMET_DESC_COLS + 
                [f'morgan_{i}' for i in range(self.settings.fingerprint_size)]
            )
            return default_features
    
    def save_model(self, filepath: Union[str, Path]) -> bool:
        """
        保存 ADMET 预测模型
        
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
                'admet_type': self.admet_type.value,
                'version': self.version,
                'task_type': self.task_type,
                'is_trained': self._is_trained,
                'scaler': self.scaler,
                'is_fitted_scaler': self._is_fitted_scaler,
                'feature_cols': self.feature_cols,
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
        加载 ADMET 预测模型
        
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
            self.admet_type = ADMETType(model_data.get('admet_type', self.admet_type.value))
            self.task_type = model_data.get('task_type', self.task_type)
            self._is_trained = model_data.get('is_trained', False)
            self.scaler = model_data.get('scaler', self.scaler)
            self._is_fitted_scaler = model_data.get('is_fitted_scaler', False)
            self.feature_cols = model_data.get('feature_cols', [])
            self.threshold = model_data.get('threshold', self.threshold)
            
            logger.info(f"模型从 {filepath} 加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False


def create_admet_predictor(admet_type: ADMETType = ADMETType.SOLUBILITY) -> ADMETPredictor:
    """
    创建 ADMET 预测器的便捷函数
    
    Args:
        admet_type: ADMET 性质类型
        
    Returns:
        ADMETPredictor 实例
    """
    return ADMETPredictor(admet_type=admet_type)