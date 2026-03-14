#!/usr/bin/env python3
"""
药学研究 AI 工作流 - 增强版ADMET预测 (修复版)
修复模型加载问题，基于成功的测试结果
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

# 化学信息学
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# 机器学习
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

# 导入CYP450预测器
from cyp450_prediction import CYP450Predictor

warnings.filterwarnings('ignore')


class EnhancedADMETPrediction:
    """增强版ADMET性质预测模块 (修复版)"""
    
    def __init__(self, output_dir: str = "./pharma_demo"):
        self.output_dir = output_dir
        self.models = {}
        self.cyp_predictors = {}
        
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        os.makedirs(f"{output_dir}/cyp450_predictions", exist_ok=True)
        
        print(f"✅ 增强版ADMET预测模块初始化")
        
        # 初始化CYP450预测器
        for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            model_path = f"models/{isoform.lower()}_model.pkl"
            if os.path.exists(model_path):
                print(f"   加载 {isoform} 预测器: {model_path}")
                try:
                    predictor = CYP450Predictor(cyp_isoform=isoform)
                    
                    # 加载模型数据 (字典格式)
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict):
                        predictor.model = model_data.get('model')
                        predictor.desc_cols = model_data.get('desc_cols')
                        predictor.feature_importance = model_data.get('feature_importance')
                        predictor.cyp_isoform = model_data.get('cyp_isoform', isoform)
                        self.cyp_predictors[isoform] = predictor
                        print(f"     ✅ {isoform} 预测器初始化成功")
                    else:
                        print(f"     ⚠️ {isoform} 模型数据结构不符")
                except Exception as e:
                    print(f"     ❌ {isoform} 初始化失败: {e}")
            else:
                print(f"   ⚠️ {isoform} 模型未找到: {model_path}")
    
    # ==================== 1. CYP450多亚型预测 ====================
    
    def predict_multiple_cyp_inhibition(self, smiles: str) -> Dict[str, Any]:
        """
        预测分子对多个CYP450亚型的抑制活性
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            dict: 各亚型抑制预测结果
        """
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
            
            # 计算综合DDI风险
            results = self._calculate_ddi_risk(results)
            
        except Exception as e:
            results['valid'] = False
            results['error'] = str(e)
        
        return results
    
    def _calculate_ddi_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算药物相互作用风险
        
        Args:
            results: 各亚型预测结果
            
        Returns:
            dict: 增加DDI风险评估
        """
        # 收集所有亚型的抑制概率
        probs = []
        for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            prob_key = f'{isoform}_probability'
            if prob_key in results and results[prob_key] is not None:
                probs.append(results[prob_key])
        
        if probs:
            # 使用最高抑制概率作为主要风险指标
            max_prob = max(probs)
            
            # 计算抑制亚型数量
            inhibitor_count = 0
            for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
                inhibitor_key = f'{isoform}_is_inhibitor'
                if inhibitor_key in results and results[inhibitor_key]:
                    inhibitor_count += 1
            
            # 风险评估逻辑
            if max_prob > 0.8:
                results['ddi_risk_level'] = '高'
                results['ddi_risk_desc'] = f'强CYP抑制 (概率: {max_prob:.2f}, {inhibitor_count}个亚型)'
            elif max_prob > 0.6:
                results['ddi_risk_level'] = '中'
                results['ddi_risk_desc'] = f'中度CYP抑制 (概率: {max_prob:.2f}, {inhibitor_count}个亚型)'
            else:
                results['ddi_risk_level'] = '低'
                results['ddi_risk_desc'] = f'低CYP抑制风险'
        else:
            results['ddi_risk_level'] = '未知'
            results['ddi_risk_desc'] = '预测数据不足'
        
        return results
    
    def batch_predict_cyp_inhibition(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        批量预测CYP450抑制
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            DataFrame: 批量预测结果
        """
        results_list = []
        
        print(f"\n🔮 批量CYP450抑制预测 ({len(smiles_list)}个分子)")
        print("=" * 60)
        
        for i, smiles in enumerate(smiles_list, 1):
            if i % 10 == 0:
                print(f"  处理中: {i}/{len(smiles_list)}")
            
            result = self.predict_multiple_cyp_inhibition(smiles)
            if result['valid']:
                results_list.append(result)
            else:
                print(f"  ⚠️  跳过无效SMILES: {smiles}")
        
        # 转换为DataFrame
        if results_list:
            df = pd.DataFrame(results_list)
            
            # 保存结果
            output_path = f"{self.output_dir}/cyp450_predictions/batch_results_{len(df)}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n✅ 批量预测完成")
            print(f"   有效分子: {len(df)}/{len(smiles_list)}")
            print(f"   结果保存: {output_path}")
            
            # 生成统计摘要
            self._generate_cyp_statistics(df)
            
            return df
        else:
            print("❌ 无有效预测结果")
            return pd.DataFrame()
    
    def _generate_cyp_statistics(self, df: pd.DataFrame):
        """生成CYP预测统计摘要"""
        print("\n📊 CYP450抑制统计:")
        print("-" * 40)
        
        for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            inhibitor_key = f'{isoform}_is_inhibitor'
            prob_key = f'{isoform}_probability'
            
            if inhibitor_key in df.columns and prob_key in df.columns:
                # 过滤有效数据
                valid_df = df[df[prob_key].notna()]
                if len(valid_df) > 0:
                    inhibitor_count = valid_df[inhibitor_key].sum()
                    avg_prob = valid_df[prob_key].mean()
                    
                    print(f"  {isoform}:")
                    print(f"    抑制剂: {inhibitor_count}/{len(valid_df)} ({inhibitor_count/len(valid_df)*100:.1f}%)")
                    print(f"    平均抑制概率: {avg_prob:.3f}")
        
        # DDI风险分布
        if 'ddi_risk_level' in df.columns:
            print("\n  DDI风险评估:")
            risk_counts = df['ddi_risk_level'].value_counts()
            for risk, count in risk_counts.items():
                percentage = count / len(df) * 100
                print(f"    {risk}: {count} ({percentage:.1f}%)")
    
    # ==================== 2. 综合报告生成 ====================
    
    def generate_comprehensive_report(self, smiles: str) -> Dict[str, Any]:
        """
        生成综合ADMET报告
        
        Args:
            smiles: 查询分子SMILES
            
        Returns:
            dict: 综合报告
        """
        print("\n📄 生成综合ADMET报告")
        print("=" * 60)
        
        # 收集所有预测结果
        cyp_results = self.predict_multiple_cyp_inhibition(smiles)
        
        # 生成报告
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'molecule': {
                'smiles': smiles,
                'valid': cyp_results['valid']
            },
            'cyp450_prediction': {
                'CYP3A4': {
                    'is_inhibitor': cyp_results.get('CYP3A4_is_inhibitor', None),
                    'probability': cyp_results.get('CYP3A4_probability', None),
                    'confidence': cyp_results.get('CYP3A4_confidence', None)
                },
                'CYP2D6': {
                    'is_inhibitor': cyp_results.get('CYP2D6_is_inhibitor', None),
                    'probability': cyp_results.get('CYP2D6_probability', None),
                    'confidence': cyp_results.get('CYP2D6_confidence', None)
                },
                'CYP2C9': {
                    'is_inhibitor': cyp_results.get('CYP2C9_is_inhibitor', None),
                    'probability': cyp_results.get('CYP2C9_probability', None),
                    'confidence': cyp_results.get('CYP2C9_confidence', None)
                }
            },
            'ddi_risk_assessment': {
                'level': cyp_results.get('ddi_risk_level', '未知'),
                'description': cyp_results.get('ddi_risk_desc', '')
            },
            'recommendations': self._generate_recommendations(cyp_results)
        }
        
        # 保存报告
        report_path = f"{self.output_dir}/comprehensive_reports/{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{hash(smiles) % 10000}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告生成完成")
        print(f"   保存路径: {report_path}")
        
        return report
    
    def _generate_recommendations(self, cyp_results: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于CYP抑制的建议
        if cyp_results.get('ddi_risk_level') == '高':
            recommendations.append("⚠️ 强CYP抑制风险，建议进行临床DDI研究")
            recommendations.append("📋 考虑调整剂量或避免联合使用其他CYP底物药物")
        elif cyp_results.get('ddi_risk_level') == '中':
            recommendations.append("⚠️ 中度CYP抑制风险，建议进行体外DDI研究")
            recommendations.append("🔬 监测相关CYP亚型的药物浓度")
        
        if not recommendations:
            recommendations.append("✅ CYP抑制风险低，常规临床监测即可")
        
        return recommendations


# ==================== 使用示例 ====================

def demo_enhanced_admet():
    """
    演示增强版ADMET预测
    """
    print("=" * 70)
    print("🧪 药学研究 AI 工作流 - 增强版ADMET预测 (修复版)")
    print("=" * 70)
    
    # 初始化
    enhanced = EnhancedADMETPrediction(output_dir="./pharma_demo")
    
    if not enhanced.cyp_predictors:
        print("❌ 无可用CYP预测器，演示终止")
        return
    
    print(f"✅ 可用CYP预测器: {list(enhanced.cyp_predictors.keys())}")
    
    # 测试分子
    test_molecules = [
        'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C',  # Simvastatin
        'CC(C)NCC(COC1=CC=C(C=C1)CC(C)N)O',     # Metoprolol
        'CC(=O)CC(C1=CC=CC=C1)C2=C(C=CC=C2)O',  # Warfarin
        'CNCCC(C1=CC=CC=C1)C2=CC=C(C=C2)OC',    # Fluoxetine
        'C1=CC=C(C=C1)C=O',                     # Benzaldehyde (简单分子)
    ]
    
    # 1. 批量CYP450预测
    print("\n" + "=" * 70)
    print("1. 批量CYP450抑制预测")
    print("=" * 70)
    
    batch_results = enhanced.batch_predict_cyp_inhibition(test_molecules)
    
    if not batch_results.empty:
        print("\n📋 批量预测结果摘要:")
        print(batch_results[['smiles', 'CYP3A4_is_inhibitor', 'CYP2D6_is_inhibitor', 
                            'CYP2C9_is_inhibitor', 'ddi_risk_level']].to_string())
    
    # 2. 单个分子详细分析
    print("\n" + "=" * 70)
    print("2. 单个分子详细分析 (Simvastatin)")
    print("=" * 70)
    
    simvastatin_smiles = 'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C'
    cyp_result = enhanced.predict_multiple_cyp_inhibition(simvastatin_smiles)
    
    if cyp_result['valid']:
        print("\nCYP450抑制预测结果:")
        for isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            print(f"  {isoform}:")
            print(f"    抑制剂: {cyp_result.get(f'{isoform}_is_inhibitor', 'N/A')}")
            prob = cyp_result.get(f'{isoform}_probability', None)
            print(f"    概率: {prob if prob is not None else 'N/A'}")
        
        print(f"\nDDI风险评估: {cyp_result.get('ddi_risk_level', 'N/A')}")
        print(f"  描述: {cyp_result.get('ddi_risk_desc', 'N/A')}")
    else:
        print(f"❌ 预测失败: {cyp_result.get('error', '未知错误')}")
    
    # 3. 综合报告
    print("\n" + "=" * 70)
    print("3. 综合ADMET报告生成")
    print("=" * 70)
    
    report = enhanced.generate_comprehensive_report(simvastatin_smiles)
    
    print("\n📋 报告摘要:")
    print(f"   分子有效性: {report['molecule']['valid']}")
    print(f"   DDI风险等级: {report['ddi_risk_assessment']['level']}")
    print(f"   风险描述: {report['ddi_risk_assessment']['description']}")
    print("\n💡 建议:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "=" * 70)
    print("✅ 增强版ADMET预测完成!")
    print("=" * 70)
    
    return enhanced, batch_results, report


if __name__ == "__main__":
    enhanced, batch_results, report = demo_enhanced_admet()