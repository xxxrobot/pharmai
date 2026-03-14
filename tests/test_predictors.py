"""
PharmaAI 预测器测试模块

测试新架构中的预测器类
"""

import unittest
import warnings
import tempfile
import os
import pandas as pd
import numpy as np
from rdkit import Chem

# 抑制弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 导入待测试的预测器
try:
    from pharmaai.predictors.cyp450 import CYP450Predictor
    from pharmaai.predictors.toxicity import ToxicityPredictor, ToxicityType
    from pharmaai.predictors.admet import ADMETPredictor, ADMETType
    PREDICTORS_AVAILABLE = True
except ImportError as e:
    print(f"无法导入预测器: {e}")
    PREDICTORS_AVAILABLE = False
    # 创建虚拟类以便测试能通过导入检查
    class CYP450Predictor:
        pass
    class ToxicityPredictor:
        pass
    class ADMETPredictor:
        pass
    class ToxicityType:
        HERG = "hERG"
    class ADMETType:
        SOLUBILITY = "Solubility"


@unittest.skipIf(not PREDICTORS_AVAILABLE, "预测器模块不可用")
class TestCYP450Predictor(unittest.TestCase):
    """CYP450 预测器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.predictor = CYP450Predictor(isoform='CYP3A4')
        
        # 创建测试数据
        self.test_smiles = [
            'CCO',  # 乙醇
            'CC(=O)O',  # 醋酸
            'c1ccccc1',  # 苯
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
        ]
        
        # 训练数据（虚拟）
        self.train_data = pd.DataFrame({
            'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
            'is_inhibitor': [0, 1, 0, 1]
        })
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.isoform, 'CYP3A4')
        self.assertEqual(self.predictor.model_name, 'cyp450_cyp3a4')
        self.assertFalse(self.predictor._is_trained)
    
    def test_molecule_validation(self):
        """测试分子验证"""
        # 有效 SMILES
        mol = self.predictor.validate_molecule('CCO')
        self.assertIsNotNone(mol)
        self.assertTrue(isinstance(mol, Chem.rdchem.Mol))
        
        # 无效 SMILES
        mol = self.predictor.validate_molecule('INVALID')
        self.assertIsNone(mol)
    
    def test_extract_features(self):
        """测试特征提取"""
        mol = Chem.MolFromSmiles('CCO')
        features = self.predictor._extract_features(mol)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_predict_untrained(self):
        """测试未训练时的预测"""
        # 未训练的预测器应该抛出异常
        with self.assertRaises(Exception):
            self.predictor.predict('CCO')
    
    def test_save_load_model(self):
        """测试模型保存和加载"""
        # 先训练一个简单模型
        self.predictor.train(self.train_data, test_size=0.5)
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = self.predictor.save_model(tmp_path)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(tmp_path))
            
            # 创建新预测器并加载模型
            new_predictor = CYP450Predictor(isoform='CYP3A4')
            result = new_predictor.load_model(tmp_path)
            self.assertTrue(result)
            self.assertTrue(new_predictor._is_trained)
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def tearDown(self):
        """测试后清理"""
        del self.predictor


@unittest.skipIf(not PREDICTORS_AVAILABLE, "预测器模块不可用")
class TestToxicityPredictor(unittest.TestCase):
    """毒性预测器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.predictor = ToxicityPredictor(toxicity_type=ToxicityType.HERG)
        
        # 测试数据
        self.test_smiles = [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.toxicity_type, ToxicityType.HERG)
        self.assertEqual(self.predictor.model_name, 'toxicity_herg')
        self.assertFalse(self.predictor._is_trained)
    
    def test_toxicity_type_enum(self):
        """测试毒性类型枚举"""
        self.assertEqual(ToxicityType.HERG.value, 'hERG')
        self.assertEqual(ToxicityType.HEPATOTOXICITY.value, 'Hepatotoxicity')
        self.assertEqual(ToxicityType.AMES.value, 'Ames')
    
    def test_smarts_patterns(self):
        """测试 SMARTS 模式初始化"""
        # 检查 SMARTS 模式字典
        self.assertIsInstance(self.predictor._smarts_patterns, dict)
        self.assertGreater(len(self.predictor._smarts_patterns), 0)
    
    def test_get_feature_names(self):
        """测试特征名称获取"""
        feature_names = self.predictor.get_feature_names()
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        
        # 检查是否包含预期特征
        self.assertIn('MW', feature_names)
        self.assertIn('LogP', feature_names)
        self.assertIn('has_basic_amine', feature_names)


@unittest.skipIf(not PREDICTORS_AVAILABLE, "预测器模块不可用")
class TestADMETPredictor(unittest.TestCase):
    """ADMET 预测器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.predictor = ADMETPredictor(admet_type=ADMETType.SOLUBILITY)
        
        # 测试数据
        self.test_smiles = [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.admet_type, ADMETType.SOLUBILITY)
        self.assertEqual(self.predictor.model_name, 'admet_solubility')
        self.assertEqual(self.predictor.task_type, 'regression')
        self.assertFalse(self.predictor._is_trained)
    
    def test_admet_type_enum(self):
        """测试 ADMET 类型枚举"""
        self.assertEqual(ADMETType.SOLUBILITY.value, 'Solubility')
        self.assertEqual(ADMETType.METABOLIC_STABILITY.value, 'MetabolicStability')
    
    def test_task_type_detection(self):
        """测试任务类型检测"""
        # 溶解度预测应该是回归任务
        solubility_predictor = ADMETPredictor(admet_type=ADMETType.SOLUBILITY)
        self.assertEqual(solubility_predictor.task_type, 'regression')
        
        # 代谢稳定性预测应该是分类任务
        metabolic_predictor = ADMETPredictor(admet_type=ADMETType.METABOLIC_STABILITY)
        self.assertEqual(metabolic_predictor.task_type, 'classification')
    
    def test_feature_extraction(self):
        """测试特征提取"""
        mol = Chem.MolFromSmiles('CCO')
        features, feature_dict = self.predictor._extract_admet_features(mol)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertIsInstance(feature_dict, dict)
        self.assertGreater(len(feature_dict), 0)
        
        # 检查是否包含关键特征
        self.assertIn('MW', feature_dict)
        self.assertIn('LogP', feature_dict)
        self.assertIn('TPSA_ratio', feature_dict)


class TestCompatibilityLayers(unittest.TestCase):
    """兼容层测试"""
    
    def test_cyp450_compatibility(self):
        """测试 CYP450 兼容层"""
        # 测试是否能够导入兼容模块
        try:
            from cyp450_prediction import CYP450Prediction
            can_import = True
        except ImportError:
            can_import = False
        
        self.assertTrue(can_import, "无法导入 CYP450 兼容模块")
    
    def test_toxicity_compatibility(self):
        """测试毒性预测兼容层"""
        # 测试是否能够导入兼容模块
        try:
            from pharma_toxicity_prediction import ToxicityPrediction
            can_import = True
        except ImportError:
            can_import = False
        
        self.assertTrue(can_import, "无法导入毒性预测兼容模块")
    
    def test_admet_compatibility(self):
        """测试 ADMET 兼容层"""
        # 测试是否能够导入兼容模块
        try:
            from pharma_admet_prediction import ADMETPrediction
            can_import = True
        except ImportError:
            can_import = False
        
        self.assertTrue(can_import, "无法导入 ADMET 兼容模块")


class TestImportExports(unittest.TestCase):
    """导入导出测试"""
    
    def test_pharmaai_init_exports(self):
        """测试 pharmaai.__init__ 导出"""
        import pharmaai
        
        # 检查核心模块是否导出
        self.assertTrue(hasattr(pharmaai, 'BasePredictor'))
        self.assertTrue(hasattr(pharmaai, 'Settings'))
        self.assertTrue(hasattr(pharmaai, 'MorganFingerprintGenerator'))
        
        # 检查预测器是否导出（如果可用）
        if PREDICTORS_AVAILABLE:
            self.assertTrue(hasattr(pharmaai, 'CYP450Predictor'))
            self.assertTrue(hasattr(pharmaai, 'ToxicityPredictor'))
            self.assertTrue(hasattr(pharmaai, 'ADMETPredictor'))
    
    def test_predictors_init_exports(self):
        """测试 predictors.__init__ 导出"""
        try:
            from pharmaai.predictors import (
                CYP450Predictor, create_cyp450_predictor,
                ToxicityPredictor, ToxicityType, create_toxicity_predictor,
                ADMETPredictor, ADMETType, create_admet_predictor
            )
            can_import = True
        except ImportError:
            can_import = False
        
        self.assertTrue(can_import, "无法从 predictors 模块导入")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCYP450Predictor))
    suite.addTests(loader.loadTestsFromTestCase(TestToxicityPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestADMETPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestCompatibilityLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestImportExports))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)