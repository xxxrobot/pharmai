#!/usr/bin/env python3
"""
CYP450预测集成测试
测试CYP450预测器与工作流集成
"""

import os
import sys
import warnings
from typing import List, Dict, Any
import pandas as pd

from rdkit import Chem
import joblib

# 导入CYP450预测器
from cyp450_prediction import CYP450Predictor

warnings.filterwarnings('ignore')


def test_cyp450_predictor():
    """测试CYP450预测器"""
    print("🧪 测试CYP450预测器")
    print("=" * 60)
    
    test_smiles = 'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C'  # Simvastatin
    print(f"测试分子: {test_smiles}")
    
    # 测试每个CYP亚型
    isoforms = ['CYP3A4', 'CYP2D6', 'CYP2C9']
    
    for isoform in isoforms:
        print(f"\n🔬 测试 {isoform}:")
        try:
            # 初始化预测器
            predictor = CYP450Predictor(cyp_isoform=isoform)
            
            # 模型文件路径
            model_file = f"models/{isoform.lower()}_model.pkl"
            
            if os.path.exists(model_file):
                print(f"  加载模型: {model_file}")
                
                # 尝试不同方法加载模型
                try:
                    # 方法1: 使用load_model方法
                    success = predictor.load_model(model_file)
                    if success:
                        print(f"  ✅ 模型加载成功")
                        
                        # 进行预测
                        result = predictor.predict(test_smiles)
                        if result and 'prediction' in result:
                            pred = result['prediction']
                            prob = result['probability']
                            conf = result['confidence']
                            
                            print(f"  预测结果: {pred} (抑制剂: {pred == 1})")
                            print(f"  预测概率: {prob}")
                            print(f"  置信度: {conf:.3f}")
                        else:
                            print(f"  ❌ 预测失败: 无效结果")
                    else:
                        print(f"  ❌ 模型加载失败")
                        
                except Exception as e:
                    print(f"  ❌ 加载方法1失败: {e}")
                    
                    # 方法2: 直接加载joblib
                    try:
                        print("  尝试直接加载joblib...")
                        model_data = joblib.load(model_file)
                        print(f"  ✅ 直接加载成功")
                        print(f"  模型类型: {type(model_data)}")
                        
                        if isinstance(model_data, dict):
                            print(f"  模型数据键: {list(model_data.keys())}")
                            predictor.model = model_data.get('model')
                            predictor.desc_cols = model_data.get('desc_cols')
                            predictor.cyp_isoform = model_data.get('cyp_isoform', isoform)
                            predictor.feature_importance = model_data.get('feature_importance')
                            
                            # 尝试预测
                            result = predictor.predict(test_smiles)
                            if result:
                                print(f"  ✅ 预测成功: {result}")
                            else:
                                print(f"  ❌ 预测失败")
                        else:
                            print(f"  ⚠️ 模型数据结构不符")
                            
                    except Exception as e2:
                        print(f"  ❌ 直接加载失败: {e2}")
                        
            else:
                print(f"  ⚠️ 模型文件不存在: {model_file}")
                
        except Exception as e:
            print(f"  ❌ {isoform} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ CYP450预测器测试完成")


def simple_enhanced_admet():
    """简化版增强ADMET预测"""
    print("\n🧪 简化版增强ADMET预测")
    print("=" * 60)
    
    # 创建简单的增强预测器
    class SimpleEnhancedADMET:
        def __init__(self):
            self.cyp_predictors = {}
            self.models_dir = "models"
            
            # 初始化CYP450预测器
            for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
                model_file = f"{self.models_dir}/{isoform.lower()}_model.pkl"
                if os.path.exists(model_file):
                    print(f"初始化 {isoform} 预测器...")
                    try:
                        predictor = CYP450Predictor(cyp_isoform=isoform)
                        # 尝试加载模型
                        if os.path.exists(model_file):
                            model_data = joblib.load(model_file)
                            if isinstance(model_data, dict) and 'model' in model_data:
                                predictor.model = model_data['model']
                                predictor.desc_cols = model_data['desc_cols']
                                predictor.feature_importance = model_data.get('feature_importance')
                                self.cyp_predictors[isoform] = predictor
                                print(f"  ✅ {isoform} 初始化成功")
                            else:
                                print(f"  ⚠️ {isoform} 模型数据结构不符")
                        else:
                            print(f"  ❌ {isoform} 模型文件不存在")
                    except Exception as e:
                        print(f"  ❌ {isoform} 初始化失败: {e}")
                else:
                    print(f"  ⚠️ {isoform} 模型文件不存在: {model_file}")
        
        def predict_cyp_inhibition(self, smiles: str) -> Dict[str, Any]:
            """预测CYP450抑制"""
            results = {'smiles': smiles, 'valid': True}
            mol = Chem.MolFromSmiles(smiles)
            
            if not mol:
                results['valid'] = False
                results['error'] = 'Invalid SMILES'
                return results
            
            try:
                # 对每个CYP亚型进行预测
                for isoform, predictor in self.cyp_predictors.items():
                    try:
                        # 使用CYP450预测器进行预测
                        pred_result = predictor.predict(smiles)
                        
                        if pred_result and isinstance(pred_result, dict):
                            results[f'{isoform}_is_inhibitor'] = pred_result.get('prediction', 0) == 1
                            probs = pred_result.get('probability', [])
                            if isinstance(probs, (list, tuple)) and len(probs) > 1:
                                results[f'{isoform}_probability'] = probs[1]
                            else:
                                results[f'{isoform}_probability'] = None
                            results[f'{isoform}_confidence'] = pred_result.get('confidence', 0)
                        else:
                            results[f'{isoform}_is_inhibitor'] = None
                            results[f'{isoform}_probability'] = None
                            results[f'{isoform}_confidence'] = None
                            
                    except Exception as e:
                        print(f"  {isoform} 预测失败: {e}")
                        results[f'{isoform}_is_inhibitor'] = None
                        results[f'{isoform}_probability'] = None
                        results[f'{isoform}_confidence'] = None
                
                # 计算DDI风险
                results = self._calculate_ddi_risk(results)
                
            except Exception as e:
                results['valid'] = False
                results['error'] = str(e)
            
            return results
        
        def _calculate_ddi_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
            """计算DDI风险"""
            # 收集所有亚型的抑制概率
            probs = []
            for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
                prob_key = f'{isoform}_probability'
                if prob_key in results and results[prob_key] is not None:
                    probs.append(results[prob_key])
            
            if probs:
                max_prob = max(probs) if probs else 0
                
                # 计算抑制亚型数量
                inhibitor_count = 0
                for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
                    inhibitor_key = f'{isoform}_is_inhibitor'
                    if inhibitor_key in results and results[inhibitor_key]:
                        inhibitor_count += 1
                
                # 风险评估
                if max_prob > 0.8:
                    results['ddi_risk_level'] = '高'
                    results['ddi_risk_desc'] = f'强CYP抑制'
                elif max_prob > 0.6:
                    results['ddi_risk_level'] = '中'
                    results['ddi_risk_desc'] = f'中度CYP抑制'
                else:
                    results['ddi_risk_level'] = '低'
                    results['ddi_risk_desc'] = f'低CYP抑制风险'
            else:
                results['ddi_risk_level'] = '未知'
                results['ddi_risk_desc'] = '预测数据不足'
            
            return results
    
    # 测试简化版
    try:
        print("\n初始化简化版ADMET预测器...")
        enhanced = SimpleEnhancedADMET()
        
        if not enhanced.cyp_predictors:
            print("❌ 无可用CYP预测器")
            return
        
        print(f"✅ 初始化完成，可用预测器: {list(enhanced.cyp_predictors.keys())}")
        
        # 测试预测
        test_smiles_list = [
            'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C',  # Simvastatin
            'CC(C)NCC(COC1=CC=C(C=C1)CC(C)N)O',     # Metoprolol
            'CC(=O)CC(C1=CC=CC=C1)C2=C(C=CC=C2)O',  # Warfarin
            'CNCCC(C1=CC=CC=C1)C2=CC=C(C=C2)OC',    # Fluoxetine
        ]
        
        print(f"\n🔮 测试 {len(test_smiles_list)} 个分子...")
        
        for i, smiles in enumerate(test_smiles_list, 1):
            print(f"\n分子 {i}: {smiles[:30]}...")
            result = enhanced.predict_cyp_inhibition(smiles)
            
            if result['valid']:
                print(f"  有效性: ✅")
                for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
                    is_inhibitor = result.get(f'{isoform}_is_inhibitor')
                    probability = result.get(f'{isoform}_probability')
                    
                    if is_inhibitor is not None:
                        print(f"  {isoform}: {'抑制剂' if is_inhibitor else '非抑制剂'} (概率: {probability:.3f})")
                    else:
                        print(f"  {isoform}: 预测失败")
                
                print(f"  DDI风险: {result.get('ddi_risk_level', '未知')}")
            else:
                print(f"  有效性: ❌ {result.get('error', '未知错误')}")
        
        print("\n" + "=" * 60)
        print("✅ 简化版增强ADMET测试完成")
        
    except Exception as e:
        print(f"❌ 简化版测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 CYP450预测集成测试")
    print("=" * 70)
    
    # 测试1: 基础CYP450预测器
    test_cyp450_predictor()
    
    # 测试2: 简化版集成
    simple_enhanced_admet()
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成!")
    print("=" * 70)