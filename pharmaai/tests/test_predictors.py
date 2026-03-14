"""
PharmaAI 单元测试

测试预测器、工具函数和配置类
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path

import pytest
import numpy as np
from rdkit import Chem

# 确保可以导入pharmaai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pharmaai.core.utils import (
    MorganFingerprintGenerator,
    calculate_molecular_features,
    prepare_features,
    validate_smiles,
    smiles_to_mol,
    batch_validate_smiles
)
from pharmaai.core.config import Settings, get_settings, reset_settings
from pharmaai.core.base_predictor import BasePredictor, SklearnPredictor, PredictionResult


# ============== Fixtures ==============

@pytest.fixture
def sample_smiles():
    """提供示例SMILES字符串"""
    return {
        'ethanol': 'CCO',
        'benzene': 'c1ccccc1',
        'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'invalid': 'invalid_smiles'
    }


@pytest.fixture
def sample_mols(sample_smiles):
    """提供示例RDKit分子对象"""
    return {
        name: Chem.MolFromSmiles(smiles) if name != 'invalid' else None
        for name, smiles in sample_smiles.items()
    }


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(autouse=True)
def reset_singletons():
    """每个测试前重置单例"""
    reset_settings()
    # 重置MorganFingerprintGenerator单例
    MorganFingerprintGenerator._instance = None
    yield


# ============== Test MorganFingerprintGenerator ==============

class TestMorganFingerprintGenerator:
    """测试Morgan指纹生成器"""
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        gen1 = MorganFingerprintGenerator.get_instance()
        gen2 = MorganFingerprintGenerator.get_instance()
        assert gen1 is gen2
    
    def test_initialization(self):
        """测试初始化参数"""
        gen = MorganFingerprintGenerator(radius=3, fp_size=1024)
        assert gen.radius == 3
        assert gen.fp_size == 1024
    
    def test_generate_from_smiles(self, sample_smiles):
        """测试从SMILES生成指纹"""
        gen = MorganFingerprintGenerator.get_instance()
        fp = gen.generate(sample_smiles['ethanol'])
        
        assert fp is not None
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 2048  # 默认大小
    
    def test_generate_from_mol(self, sample_mols):
        """测试从分子对象生成指纹"""
        gen = MorganFingerprintGenerator.get_instance()
        fp = gen.generate(sample_mols['benzene'])
        
        assert fp is not None
        assert isinstance(fp, np.ndarray)
    
    def test_generate_invalid_molecule(self):
        """测试无效分子处理"""
        gen = MorganFingerprintGenerator.get_instance()
        fp = gen.generate('invalid_smiles')
        assert fp is None
    
    def test_generate_bulk(self, sample_smiles):
        """测试批量生成指纹"""
        gen = MorganFingerprintGenerator.get_instance()
        smiles_list = [
            sample_smiles['ethanol'],
            sample_smiles['benzene'],
            sample_smiles['aspirin']
        ]
        fps = gen.generate_bulk(smiles_list)
        
        assert len(fps) == 3
        assert all(isinstance(fp, np.ndarray) for fp in fps)
    
    def test_get_feature_names(self):
        """测试获取特征名称"""
        gen = MorganFingerprintGenerator.get_instance()
        names = gen.get_feature_names()
        
        assert len(names) == 2048
        assert names[0] == 'morgan_0'
        assert names[-1] == 'morgan_2047'


# ============== Test calculate_molecular_features ==============

class TestCalculateMolecularFeatures:
    """测试分子特征计算"""
    
    def test_basic_features(self, sample_smiles):
        """测试基本特征计算"""
        features = calculate_molecular_features(sample_smiles['ethanol'])
        
        assert features is not None
        assert 'MW' in features
        assert 'LogP' in features
        assert 'TPSA' in features
        assert 'HBD' in features
        assert 'HBA' in features
    
    def test_ethanol_properties(self, sample_smiles):
        """测试乙醇的特定属性"""
        features = calculate_molecular_features(sample_smiles['ethanol'])
        
        # 乙醇分子量约为46
        assert 45 < features['MW'] < 47
        # 乙醇有1个氢键供体（OH）
        assert features['HBD'] == 1
        # 乙醇有1个氢键受体（O）
        assert features['HBA'] == 1
    
    def test_aspirin_properties(self, sample_smiles):
        """测试阿司匹林的特定属性"""
        features = calculate_molecular_features(sample_smiles['aspirin'])
        
        # 阿司匹林有苯环
        assert features['AromaticRings'] >= 1
        # 阿司匹林有多个可旋转键
        assert features['RotatableBonds'] >= 2
    
    def test_with_fingerprint(self, sample_smiles):
        """测试包含指纹的特征"""
        features = calculate_molecular_features(
            sample_smiles['benzene'],
            include_fingerprint=True
        )
        
        assert 'fingerprint' in features
        assert isinstance(features['fingerprint'], np.ndarray)
        assert len(features['fingerprint']) == 2048
    
    def test_invalid_smiles(self):
        """测试无效SMILES处理"""
        features = calculate_molecular_features('not_a_valid_smiles')
        assert features is None
    
    def test_from_mol_object(self, sample_mols):
        """测试从分子对象计算"""
        features = calculate_molecular_features(sample_mols['caffeine'])
        assert features is not None
        assert features['MW'] > 0


# ============== Test validate_smiles ==============

class TestValidateSmiles:
    """测试SMILES验证"""
    
    def test_valid_smiles(self, sample_smiles):
        """测试有效SMILES"""
        assert validate_smiles(sample_smiles['ethanol']) is True
        assert validate_smiles(sample_smiles['benzene']) is True
    
    def test_invalid_smiles(self):
        """测试无效SMILES"""
        assert validate_smiles('invalid') is False
        assert validate_smiles('C1C') is False  # 不完整的环
    
    def test_empty_string(self):
        """测试空字符串"""
        assert validate_smiles('') is False


# ============== Test smiles_to_mol ==============

class TestSmilesToMol:
    """测试SMILES转换"""
    
    def test_valid_conversion(self, sample_smiles):
        """测试有效转换"""
        mol = smiles_to_mol(sample_smiles['ethanol'])
        assert mol is not None
        assert isinstance(mol, Chem.rdchem.Mol)
    
    def test_invalid_conversion(self):
        """测试无效转换"""
        mol = smiles_to_mol('invalid')
        assert mol is None


# ============== Test batch_validate_smiles ==============

class TestBatchValidateSmiles:
    """测试批量SMILES验证"""
    
    def test_batch_validation(self, sample_smiles):
        """测试批量验证"""
        smiles_list = [
            sample_smiles['ethanol'],
            'invalid',
            sample_smiles['benzene'],
            'also_invalid'
        ]
        valid_smiles, valid_indices = batch_validate_smiles(smiles_list)
        
        assert len(valid_smiles) == 2
        assert valid_indices == [0, 2]
        assert sample_smiles['ethanol'] in valid_smiles
        assert sample_smiles['benzene'] in valid_smiles


# ============== Test Settings ==============

class TestSettings:
    """测试配置类"""
    
    def test_default_values(self):
        """测试默认值"""
        settings = Settings()
        
        assert settings.model_path == './models'
        assert settings.data_path == './data'
        assert settings.log_level == 'INFO'
        assert settings.fingerprint_radius == 2
        assert settings.fingerprint_size == 2048
    
    def test_environment_variables(self, monkeypatch, temp_dir):
        """测试环境变量覆盖"""
        monkeypatch.setenv('PHARMAAI_MODEL_PATH', '/custom/models')
        monkeypatch.setenv('PHARMAAI_LOG_LEVEL', 'DEBUG')
        
        reset_settings()
        settings = get_settings()
        
        assert settings.model_path == '/custom/models'
        assert settings.log_level == 'DEBUG'
    
    def test_get_model_path(self):
        """测试获取模型路径"""
        settings = Settings()
        path = settings.get_model_path('test_model')
        assert path.endswith('test_model.joblib')
    
    def test_get_data_path(self):
        """测试获取数据路径"""
        settings = Settings()
        path = settings.get_data_path('data.csv')
        assert path.endswith('data.csv')
    
    def test_to_dict(self):
        """测试转换为字典"""
        settings = Settings()
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model_path' in config_dict
        assert 'cyp450_isoforms' in config_dict
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


# ============== Test PredictionResult ==============

class TestPredictionResult:
    """测试预测结果类"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        result = PredictionResult(value=0.8, confidence=0.95)
        assert result.value == 0.8
        assert result.confidence == 0.95
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = PredictionResult(
            value=True,
            confidence=0.9,
            model_name='test_model',
            metadata={'key': 'value'}
        )
        d = result.to_dict()
        assert d['value'] is True
        assert d['confidence'] == 0.9
        assert d['model_name'] == 'test_model'
    
    def test_invalid_confidence(self, caplog):
        """测试无效置信度警告"""
        caplog.set_level(logging.WARNING)
        result = PredictionResult(value=1, confidence=1.5)
        assert 'outside' in caplog.text


# ============== Test BasePredictor ==============

class TestBasePredictor:
    """测试基础预测器抽象类"""
    
    def test_cannot_instantiate(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            BasePredictor()
    
    def test_validate_molecule(self, sample_smiles):
        """测试分子验证"""
        class DummyPredictor(BasePredictor):
            def predict(self, mol):
                pass
            def batch_predict(self, molecules):
                pass
            def save_model(self, filepath):
                pass
            def load_model(self, filepath):
                pass
        
        predictor = DummyPredictor()
        
        # 有效SMILES
        mol = predictor.validate_molecule(sample_smiles['ethanol'])
        assert mol is not None
        
        # 无效SMILES
        mol = predictor.validate_molecule('invalid')
        assert mol is None
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        class DummyPredictor(BasePredictor):
            def predict(self, mol):
                pass
            def batch_predict(self, molecules):
                pass
            def save_model(self, filepath):
                pass
            def load_model(self, filepath):
                pass
        
        predictor = DummyPredictor(model_name='test', version='2.0')
        info = predictor.get_model_info()
        
        assert info['model_name'] == 'test'
        assert info['version'] == '2.0'
        assert info['is_trained'] is False


# ============== Main ==============

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
