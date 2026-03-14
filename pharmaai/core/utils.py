"""
PharmaAI 公共工具模块

提供统一的分子指纹生成、描述符计算和特征准备功能
避免在多个文件中重复初始化
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, rdFingerprintGenerator

logger = logging.getLogger(__name__)


@dataclass
class MolecularFeatures:
    """分子特征数据类"""
    mw: float  # 分子量
    logp: float  # 脂水分配系数
    tpsa: float  # 拓扑极性表面积
    hbd: int  # 氢键供体数
    hba: int  # 氢键受体数
    rotatable_bonds: int  # 可旋转键数
    aromatic_rings: int  # 芳香环数
    heavy_atoms: int  # 重原子数
    num_nitrogens: int  # 氮原子数
    num_halogens: int  # 卤素原子数
    num_sulfurs: int  # 硫原子数
    num_oxygens: int  # 氧原子数
    num_rings: int  # 环数
    num_heteroatoms: int  # 杂原子数
    bertz_ct: float  # 分子复杂度
    mol_mr: float  # 摩尔折射率


class MorganFingerprintGenerator:
    """
    Morgan指纹生成器 (单例模式)
    
    避免在多个文件中重复初始化指纹生成器，提高性能
    
    Usage:
        >>> from pharmaai.core.utils import MorganFingerprintGenerator
        >>> fp_gen = MorganFingerprintGenerator.get_instance()
        >>> fp = fp_gen.generate(smiles)
    """
    
    _instance = None
    
    def __new__(cls, radius: int = 2, fp_size: int = 2048):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, radius: int = 2, fp_size: int = 2048):
        if self._initialized:
            return
        self.radius = radius
        self.fp_size = fp_size
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
        self._initialized = True
        logger.debug(f"MorganFingerprintGenerator initialized (radius={radius}, fpSize={fp_size})")
    
    @classmethod
    def get_instance(cls, radius: int = 2, fp_size: int = 2048) -> "MorganFingerprintGenerator":
        """获取单例实例"""
        return cls(radius, fp_size)
    
    def generate(self, mol: Union[str, Chem.rdchem.Mol]) -> Optional[np.ndarray]:
        """
        生成分子的Morgan指纹
        
        Args:
            mol: SMILES字符串或RDKit分子对象
            
        Returns:
            numpy数组形式的指纹，如果分子无效则返回None
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            logger.warning("Invalid molecule provided for fingerprint generation")
            return None
        
        fp = self._generator.GetFingerprint(mol)
        return np.array(fp)
    
    def generate_bulk(self, molecules: List[Union[str, Chem.rdchem.Mol]]) -> List[Optional[np.ndarray]]:
        """
        批量生成分子指纹
        
        Args:
            molecules: SMILES字符串或RDKit分子对象列表
            
        Returns:
            指纹数组列表
        """
        return [self.generate(mol) for mol in molecules]
    
    def get_feature_names(self) -> List[str]:
        """获取指纹特征名称列表"""
        return [f"morgan_{i}" for i in range(self.fp_size)]


def calculate_molecular_features(
    mol: Union[str, Chem.rdchem.Mol],
    include_fingerprint: bool = False,
    fp_generator: Optional[MorganFingerprintGenerator] = None
) -> Optional[Dict[str, Any]]:
    """
    计算分子描述符特征
    
    Args:
        mol: SMILES字符串或RDKit分子对象
        include_fingerprint: 是否包含Morgan指纹
        fp_generator: 指纹生成器实例（如果include_fingerprint为True）
        
    Returns:
        特征字典，如果分子无效则返回None
        
    Example:
        >>> features = calculate_molecular_features("CCO")
        >>> print(features['MW'])
        46.07
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    
    if mol is None:
        logger.warning("Invalid molecule provided for feature calculation")
        return None
    
    try:
        features = {
            # 物理化学性质
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            
            # 结构特征
            'RotatableBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'NumRings': Lipinski.RingCount(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
            
            # 原子计数
            'NumNitrogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
            'NumHalogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]),
            'NumSulfurs': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
            'NumOxygens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
            
            # 分子复杂度
            'BertzCT': GraphDescriptors.BertzCT(mol),
            'MolMR': Crippen.MolMR(mol),
        }
        
        if include_fingerprint:
            if fp_generator is None:
                fp_generator = MorganFingerprintGenerator.get_instance()
            fp = fp_generator.generate(mol)
            if fp is not None:
                features['fingerprint'] = fp
        
        return features
        
    except Exception as e:
        logger.error(f"Error calculating molecular features: {e}")
        return None


def prepare_features(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    target_col: Optional[str] = None,
    include_fingerprint: bool = True,
    fp_generator: Optional[MorganFingerprintGenerator] = None,
    desc_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    准备特征矩阵 (通用特征准备函数)
    
    Args:
        df: 包含分子数据的DataFrame
        smiles_col: SMILES列名
        target_col: 目标变量列名（可选）
        include_fingerprint: 是否包含Morgan指纹
        fp_generator: 指纹生成器实例
        desc_cols: 要包含的描述符列列表（None则使用默认列表）
        
    Returns:
        tuple: (X, y, feature_names)
            - X: 特征矩阵
            - y: 目标变量（如果target_col为None则为None）
            - feature_names: 特征名称列表
            
    Example:
        >>> X, y, names = prepare_features(df, target_col='is_toxic')
        >>> print(X.shape)
        (100, 2060)
    """
    if fp_generator is None and include_fingerprint:
        fp_generator = MorganFingerprintGenerator.get_instance()
    
    if desc_cols is None:
        desc_cols = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 
            'RotatableBonds', 'AromaticRings', 'HeavyAtoms',
            'NumNitrogens', 'NumHalogens', 'NumSulfurs', 'NumOxygens'
        ]
    
    X_desc_list = []
    X_fp_list = []
    y_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        features = calculate_molecular_features(
            row[smiles_col],
            include_fingerprint=include_fingerprint,
            fp_generator=fp_generator
        )
        
        if features is not None:
            # 提取描述符
            X_desc = [features.get(col, 0) for col in desc_cols]
            X_desc_list.append(X_desc)
            
            # 提取指纹
            if include_fingerprint and 'fingerprint' in features:
                X_fp_list.append(features['fingerprint'])
            
            # 提取目标变量
            if target_col:
                y_list.append(row[target_col])
            
            valid_indices.append(idx)
    
    if not X_desc_list:
        logger.warning("No valid molecules found in dataset")
        return np.array([]), None, []
    
    # 构建特征矩阵
    X_desc_arr = np.array(X_desc_list)
    
    if include_fingerprint and X_fp_list:
        X_fp_arr = np.array(X_fp_list)
        X = np.hstack([X_desc_arr, X_fp_arr])
        feature_names = desc_cols + fp_generator.get_feature_names()
    else:
        X = X_desc_arr
        feature_names = desc_cols
    
    y = np.array(y_list) if y_list else None
    
    logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, feature_names


def validate_smiles(smiles: str) -> bool:
    """
    验证SMILES字符串是否有效
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        是否有效
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def smiles_to_mol(smiles: str) -> Optional[Chem.rdchem.Mol]:
    """
    将SMILES转换为RDKit分子对象
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        RDKit分子对象或None
    """
    return Chem.MolFromSmiles(smiles)


def batch_validate_smiles(smiles_list: List[str]) -> Tuple[List[str], List[int]]:
    """
    批量验证SMILES字符串
    
    Args:
        smiles_list: SMILES字符串列表
        
    Returns:
        tuple: (valid_smiles, valid_indices)
    """
    valid_smiles = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        if validate_smiles(smiles):
            valid_smiles.append(smiles)
            valid_indices.append(i)
    
    return valid_smiles, valid_indices
