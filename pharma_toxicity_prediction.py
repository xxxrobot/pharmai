#!/usr/bin/env python3
"""
药学研究 AI 工作流 - 阶段二：毒性预测模块
包含：
1. hERG 心脏毒性预测
2. 肝毒性预测
3. 致突变性 (Ames) 预测
4. 综合毒性风险评估
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# 化学信息学
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
from rdkit.Chem import PandasTools, Crippen, Lipinski, Fragments, GraphDescriptors
from rdkit.Chem import rdFingerprintGenerator

# 机器学习
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

# 初始化Morgan指纹生成器 (避免弃用警告)
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


warnings.filterwarnings('ignore')


class ToxicityPrediction:
    """毒性预测模块"""
    
    def __init__(self, output_dir: str = "./pharma_demo"):
        self.output_dir = output_dir
        self.models = {}
        
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        print(f"✅ 毒性预测模块初始化")
    
    # ==================== 1. 毒性特征工程 ====================
    
    def calculate_toxicity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算毒性相关分子特征
        """
        print("\n🔬 计算毒性相关特征...")
        
        def get_toxicity_descriptors(mol):
            if mol is None:
                return {}
            
            # 基础描述符
            desc = {
                # 物理化学性质
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                
                # 结构特征
                'NumRings': Lipinski.RingCount(mol),
                'NumAromaticRings': Lipinski.NumAromaticRings(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
                
                # 毒性相关特征
                'NumNitrogens': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
                'NumSulfurs': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16),
                'NumHalogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]),
                
                # 官能团计数
                'NumAliphaticCarbocycles': Lipinski.NumAliphaticCarbocycles(mol),
                'NumAliphaticHeterocycles': Lipinski.NumAliphaticHeterocycles(mol),
                'NumAromaticCarbocycles': Lipinski.NumAromaticCarbocycles(mol),
                'NumAromaticHeterocycles': Lipinski.NumAromaticHeterocycles(mol),
                
                # 分子复杂度
                'BertzCT': GraphDescriptors.BertzCT(mol),
                'MolLogP': Crippen.MolLogP(mol),
                'MolMR': Crippen.MolMR(mol),
            }
            
            # 特定毒性警示结构
            # hERG 相关：含氮杂环、芳香胺等
            desc['has_basic_amine'] = self._has_basic_amine(mol)
            desc['has_aromatic_amine'] = self._has_aromatic_amine(mol)
            desc['has_piperazine'] = self._has_piperazine(mol)
            desc['has_quaternary_n'] = self._has_quaternary_n(mol)
            
            # 肝毒性相关：硝基、卤代芳烃等
            desc['has_nitro'] = self._has_nitro(mol)
            desc['has_halogenated_aromatic'] = self._has_halogenated_aromatic(mol)
            desc['has_hydrazine'] = self._has_hydrazine(mol)
            
            # 致突变性相关：烷化剂、芳香胺等
            desc['has_alkyl_halide'] = self._has_alkyl_halide(mol)
            desc['has_epoxide'] = self._has_epoxide(mol)
            desc['has_azo'] = self._has_azo(mol)
            
            return desc
        
        # 应用特征计算
        desc_df = df['mol'].apply(get_toxicity_descriptors).apply(pd.Series)
        df = pd.concat([df, desc_df], axis=1)
        
        print(f"✅ 计算完成: {len(desc_df.columns)} 个毒性特征")
        return df
    
    # 警示结构检测函数
    def _has_basic_amine(self, mol):
        """检测碱性胺基"""
        pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_aromatic_amine(self, mol):
        """检测芳香胺"""
        pattern = Chem.MolFromSmarts('[NX3;H2,H1]c1ccccc1')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_piperazine(self, mol):
        """检测哌嗪环"""
        pattern = Chem.MolFromSmarts('C1CNCCN1')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_quaternary_n(self, mol):
        """检测季铵盐"""
        pattern = Chem.MolFromSmarts('[N+;X4]')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_nitro(self, mol):
        """检测硝基"""
        pattern = Chem.MolFromSmarts('[N+](=O)[O-]')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_halogenated_aromatic(self, mol):
        """检测卤代芳烃"""
        pattern = Chem.MolFromSmarts('[F,Cl,Br,I]c1ccccc1')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_hydrazine(self, mol):
        """检测肼基"""
        pattern = Chem.MolFromSmarts('NN')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_alkyl_halide(self, mol):
        """检测烷基卤化物"""
        pattern = Chem.MolFromSmarts('[C][F,Cl,Br,I]')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_epoxide(self, mol):
        """检测环氧基"""
        pattern = Chem.MolFromSmarts('C1OC1')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_azo(self, mol):
        """检测偶氮基"""
        pattern = Chem.MolFromSmarts('N=N')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    # ==================== 2. 创建示例毒性数据集 ====================
    
    def create_toxicity_dataset(self) -> pd.DataFrame:
        """
        创建毒性预测示例数据集
        """
        print("\n📦 创建毒性数据集...")
        
        # hERG 抑制剂 (已知的心脏毒性药物)
        herg_positive = [
            ('CC(C)NCC(COc1ccccc1)O', 'hERG_positive'),  # 普萘洛尔
            ('COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1', 'hERG_positive'),  # 奥氮平
            ('c1ccc2c(c1)c(c[nH]2)CCN', 'hERG_positive'),  # 色胺
            ('CN1CCC[C@H]1c2cccnc2', 'hERG_positive'),  # 尼古丁
        ]
        
        # hERG 阴性 (安全药物)
        herg_negative = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'hERG_negative'),  # 布洛芬
            ('CC(=O)Oc1ccccc1C(=O)O', 'hERG_negative'),  # 阿司匹林
            ('c1ccc(cc1)C(=O)O', 'hERG_negative'),  # 苯甲酸
            ('CCO', 'hERG_negative'),  # 乙醇
        ]
        
        # 肝毒性阳性
        hepatotoxic_positive = [
            ('CC(=O)Nc1ccc(O)cc1', 'hepatotoxic'),  # 对乙酰氨基酚
            ('CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc31', 'hepatotoxic'),  # 地西泮类似物
        ]
        
        # 肝毒性阴性
        hepatotoxic_negative = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'safe'),
            ('CC(=O)Oc1ccccc1C(=O)O', 'safe'),
        ]
        
        # Ames 阳性 (致突变)
        ames_positive = [
            ('O=[N+]([O-])c1ccccc1', 'ames_positive'),  # 硝基苯
            ('Nc1ccccc1', 'ames_positive'),  # 苯胺
        ]
        
        # Ames 阴性
        ames_negative = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'ames_negative'),
            ('CC(=O)Oc1ccccc1C(=O)O', 'ames_negative'),
        ]
        
        # 合并数据
        all_data = []
        
        # hERG 数据
        for smi, label in herg_positive + herg_negative:
            all_data.append({
                'smiles': smi,
                'hERG': 1 if 'positive' in label else 0,
                'hepatotoxic': 0,
                'ames': 0
            })
        
        # 肝毒性数据
        for smi, label in hepatotoxic_positive + hepatotoxic_negative:
            all_data.append({
                'smiles': smi,
                'hERG': 0,
                'hepatotoxic': 1 if label == 'hepatotoxic' else 0,
                'ames': 0
            })
        
        # Ames 数据
        for smi, label in ames_positive + ames_negative:
            all_data.append({
                'smiles': smi,
                'hERG': 0,
                'hepatotoxic': 0,
                'ames': 1 if 'positive' in label else 0
            })
        
        df = pd.DataFrame(all_data)
        
        # 去重
        df = df.drop_duplicates(subset=['smiles'])
        
        # 验证SMILES
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 创建数据集: {len(df)} 个分子")
        print(f"   hERG阳性: {df['hERG'].sum()}")
        print(f"   肝毒性阳性: {df['hepatotoxic'].sum()}")
        print(f"   Ames阳性: {df['ames'].sum()}")
        
        return df
    
    # ==================== 3. 训练毒性预测模型 ====================
    
    def train_toxicity_model(self, df: pd.DataFrame, 
                            target_col: str,
                            model_type: str = 'rf') -> Dict:
        """
        训练毒性预测模型
        
        target_col: 'hERG', 'hepatotoxic', 'ames'
        model_type: 'rf' (随机森林), 'gb' (梯度提升)
        """
        print(f"\n🤖 训练 {target_col} 毒性预测模型...")
        
        # 准备特征
        feature_cols = [col for col in df.columns if col not in 
                       ['smiles', 'mol', 'hERG', 'hepatotoxic', 'ames']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 检查类别分布
        class_counts = pd.Series(y).value_counts()
        print(f"   类别分布: {dict(class_counts)}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 选择模型
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:  # gradient boosting
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
        }
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        # 特征重要性
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 保存模型
        model_path = f"{self.output_dir}/models/{target_col}_toxicity_model.pkl"
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, model_path)
        
        self.models[target_col] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }
        
        print(f"✅ 训练完成!")
        print(f"   准确率: {metrics['accuracy']:.3f}")
        print(f"   F1分数: {metrics['f1']:.3f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"   CV ROC-AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
        print(f"   Top 5 特征: {[f[0] for f in top_features]}")
        print(f"   模型保存: {model_path}")
        
        return metrics
    
    # ==================== 4. 毒性预测 ====================
    
    def predict_toxicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测分子毒性
        """
        print(f"\n🔮 预测毒性...")
        
        # 确保有特征
        if 'MW' not in df.columns:
            df = self.calculate_toxicity_features(df)
        
        # 对每个毒性终点进行预测
        for target in ['hERG', 'hepatotoxic', 'ames']:
            if target in self.models:
                model_info = self.models[target]
                model = model_info['model']
                feature_cols = model_info['feature_cols']
                
                X = df[feature_cols].values
                
                # 预测
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
                
                df[f'{target}_prediction'] = predictions
                df[f'{target}_probability'] = probabilities
                
                # 风险等级
                def get_risk_level(prob):
                    if prob >= 0.7:
                        return 'High'
                    elif prob >= 0.3:
                        return 'Medium'
                    else:
                        return 'Low'
                
                df[f'{target}_risk'] = df[f'{target}_probability'].apply(get_risk_level)
        
        print(f"✅ 毒性预测完成")
        return df
    
    # ==================== 5. 综合毒性风险评估 ====================
    
    def calculate_overall_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合毒性风险评分
        """
        print(f"\n⚠️  计算综合毒性风险...")
        
        # 风险权重 (可根据重要性调整)
        weights = {
            'hERG': 0.4,      # 心脏毒性权重高
            'hepatotoxic': 0.35,  # 肝毒性
            'ames': 0.25      # 致突变性
        }
        
        # 计算加权风险评分
        risk_score = 0
        for target, weight in weights.items():
            prob_col = f'{target}_probability'
            if prob_col in df.columns:
                risk_score += df[prob_col] * weight
        
        df['overall_toxicity_score'] = risk_score
        
        # 综合风险等级
        def get_overall_risk(score):
            if score >= 0.6:
                return 'High Risk'
            elif score >= 0.3:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        df['overall_toxicity_risk'] = df['overall_toxicity_score'].apply(get_overall_risk)
        
        # 统计
        risk_counts = df['overall_toxicity_risk'].value_counts()
        print(f"\n📊 综合风险评估:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    # ==================== 6. 生成毒性报告 ====================
    
    def generate_toxicity_report(self, df: pd.DataFrame, filename: str = "toxicity_report.json"):
        """
        生成毒性预测报告
        """
        print(f"\n📄 生成毒性报告...")
        
        report = {
            'summary': {
                'total_molecules': len(df),
                'high_risk': int((df['overall_toxicity_risk'] == 'High Risk').sum()),
                'medium_risk': int((df['overall_toxicity_risk'] == 'Medium Risk').sum()),
                'low_risk': int((df['overall_toxicity_risk'] == 'Low Risk').sum()),
            },
            'model_performance': {},
            'top_risk_molecules': []
        }
        
        # 模型性能
        for target, model_info in self.models.items():
            report['model_performance'][target] = model_info['metrics']
        
        # 高风险分子
        high_risk_df = df[df['overall_toxicity_risk'] == 'High Risk'].nlargest(5, 'overall_toxicity_score')
        for _, row in high_risk_df.iterrows():
            report['top_risk_molecules'].append({
                'smiles': row['smiles'],
                'overall_score': float(row['overall_toxicity_score']),
                'hERG_prob': float(row.get('hERG_probability', 0)),
                'hepatotoxic_prob': float(row.get('hepatotoxic_probability', 0)),
                'ames_prob': float(row.get('ames_probability', 0))
            })
        
        # 保存报告
        report_path = f"{self.output_dir}/results/{filename}"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ 报告保存: {report_path}")
        
        # 保存详细结果
        result_path = f"{self.output_dir}/results/toxicity_predictions.csv"
        df.to_csv(result_path, index=False)
        print(f"✅ 详细结果保存: {result_path}")
        
        return report


# ==================== 使用示例 ====================

def demo_toxicity_prediction():
    """
    演示毒性预测完整流程
    """
    print("=" * 70)
    print("🧪 药学研究 AI 工作流 - 阶段二：毒性预测模块")
    print("=" * 70)
    
    # 初始化
    tox = ToxicityPrediction(output_dir="./pharma_demo")
    
    # 步骤 1: 创建毒性数据集
    print("\n" + "=" * 70)
    print("步骤 1: 创建毒性数据集")
    print("=" * 70)
    
    df = tox.create_toxicity_dataset()
    
    # 步骤 2: 计算毒性特征
    print("\n" + "=" * 70)
    print("步骤 2: 计算毒性特征")
    print("=" * 70)
    
    df = tox.calculate_toxicity_features(df)
    print(f"\n特征示例:")
    print(df[['smiles', 'MW', 'LogP', 'has_basic_amine', 'has_nitro']].head().to_string())
    
    # 步骤 3: 训练毒性预测模型
    print("\n" + "=" * 70)
    print("步骤 3: 训练毒性预测模型")
    print("=" * 70)
    
    # hERG 模型
    herg_metrics = tox.train_toxicity_model(df, target_col='hERG', model_type='rf')
    
    # 肝毒性模型
    hep_metrics = tox.train_toxicity_model(df, target_col='hepatotoxic', model_type='rf')
    
    # Ames 模型
    ames_metrics = tox.train_toxicity_model(df, target_col='ames', model_type='rf')
    
    # 步骤 4: 预测毒性
    print("\n" + "=" * 70)
    print("步骤 4: 预测毒性")
    print("=" * 70)
    
    df = tox.predict_toxicity(df)
    
    # 显示预测结果
    print(f"\n预测结果示例:")
    display_cols = ['smiles', 'hERG_probability', 'hepatotoxic_probability', 
                   'ames_probability']
    print(df[display_cols].head().to_string())
    
    # 步骤 5: 综合风险评估
    print("\n" + "=" * 70)
    print("步骤 5: 综合毒性风险评估")
    print("=" * 70)
    
    df = tox.calculate_overall_risk(df)
    
    # 步骤 6: 生成报告
    print("\n" + "=" * 70)
    print("步骤 6: 生成毒性报告")
    print("=" * 70)
    
    report = tox.generate_toxicity_report(df)
    
    print("\n" + "=" * 70)
    print("✅ 毒性预测模块完成!")
    print(f"📊 总分子数: {report['summary']['total_molecules']}")
    print(f"⚠️  高风险: {report['summary']['high_risk']}")
    print(f"⚡ 中风险: {report['summary']['medium_risk']}")
    print(f"✓  低风险: {report['summary']['low_risk']}")
    print("=" * 70)
    
    return tox, df, report


if __name__ == "__main__":
    tox, df, report = demo_toxicity_prediction()
