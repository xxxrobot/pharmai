#!/usr/bin/env python3
"""
药学研究 AI 工作流 - 阶段二（续）：溶解度与代谢稳定性预测
包含：
1. 水溶性 (Solubility) 预测
2. 代谢稳定性 (Metabolic Stability) 预测
3. CYP450 抑制预测
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
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# 机器学习
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

warnings.filterwarnings('ignore')


class ADMETPrediction:
    """ADMET性质预测模块（溶解度、代谢稳定性）"""
    
    def __init__(self, output_dir: str = "./pharma_demo"):
        self.output_dir = output_dir
        self.models = {}
        
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        
        print(f"✅ ADMET预测模块初始化")
    
    # ==================== 1. 溶解度预测 ====================
    
    def calculate_solubility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算溶解度相关特征
        """
        print("\n🔬 计算溶解度特征...")
        
        def get_solubility_descriptors(mol):
            if mol is None:
                return {}
            
            # 基础物理化学性质
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # 氢键
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            # 极性表面积占比
            tpsa_ratio = tpsa / mw if mw > 0 else 0
            
            # 可旋转键
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            rot_bonds_ratio = rot_bonds / mw * 100 if mw > 0 else 0
            
            # 芳香性
            aromatic_rings = Lipinski.NumAromaticRings(mol)
            aromatic_ratio = aromatic_rings / Lipinski.RingCount(mol) if Lipinski.RingCount(mol) > 0 else 0
            
            # 电荷相关
            mol_refractivity = Crippen.MolMR(mol)
            
            # 亲水性/疏水性平衡
            hydrophilic_factor = tpsa / (logp + 1) if logp > -1 else tpsa
            
            return {
                'MW': mw,
                'LogP': logp,
                'TPSA': tpsa,
                'TPSA_ratio': tpsa_ratio,
                'HBD': hbd,
                'HBA': hba,
                'HB_total': hbd + hba,
                'RotatableBonds': rot_bonds,
                'RotBonds_ratio': rot_bonds_ratio,
                'AromaticRings': aromatic_rings,
                'Aromatic_ratio': aromatic_ratio,
                'MolRefractivity': mol_refractivity,
                'HydrophilicFactor': hydrophilic_factor,
                # 溶解度相关组合特征
                'LogS_estimated': 0.5 - 0.01 * (mw - 100) - 0.5 * logp,  # 简化估算
                'Polarity_index': tpsa / (hbd + hba + 1),
            }
        
        desc_df = df['mol'].apply(get_solubility_descriptors).apply(pd.Series)
        df = pd.concat([df, desc_df], axis=1)
        
        print(f"✅ 计算完成: {len(desc_df.columns)} 个溶解度特征")
        return df
    
    def create_solubility_dataset(self) -> pd.DataFrame:
        """
        创建溶解度示例数据集
        基于已知药物的溶解度数据 (LogS, mol/L)
        """
        print("\n📦 创建溶解度数据集...")
        
        # 高溶解度药物 (LogS > -2)
        high_solubility = [
            ('CC(=O)Oc1ccccc1C(=O)O', -1.2, 'high'),  # 阿司匹林
            ('c1ccc(cc1)C(=O)O', -1.1, 'high'),  # 苯甲酸
            ('CC(=O)O', 0.3, 'high'),  # 乙酸
            ('c1ccccc1O', -0.8, 'high'),  # 苯酚
        ]
        
        # 中等溶解度 (LogS -2 to -4)
        medium_solubility = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', -3.2, 'medium'),  # 布洛芬
            ('CC(C)NCC(COc1ccccc1)O', -2.5, 'medium'),  # 普萘洛尔
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', -2.8, 'medium'),  # 咖啡因
        ]
        
        # 低溶解度 (LogS < -4)
        low_solubility = [
            ('COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1', -4.5, 'low'),  # 奥氮平
            ('c1ccc2c(c1)c(c[nH]2)CCN', -4.8, 'low'),  # 色胺
            ('CN1CCC[C@H]1c2cccnc2', -4.2, 'low'),  # 尼古丁
        ]
        
        all_data = high_solubility + medium_solubility + low_solubility
        
        df = pd.DataFrame(all_data, columns=['smiles', 'LogS', 'solubility_class'])
        
        # 验证SMILES
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 创建数据集: {len(df)} 个分子")
        print(f"   高溶解度: {sum(df['solubility_class'] == 'high')}")
        print(f"   中溶解度: {sum(df['solubility_class'] == 'medium')}")
        print(f"   低溶解度: {sum(df['solubility_class'] == 'low')}")
        
        return df
    
    def train_solubility_model(self, df: pd.DataFrame) -> Dict:
        """
        训练溶解度预测模型 (回归模型)
        """
        print("\n🤖 训练溶解度预测模型...")
        
        # 准备特征
        feature_cols = [col for col in df.columns if col not in 
                       ['smiles', 'mol', 'LogS', 'solubility_class']]
        
        X = df[feature_cols].values
        y = df['LogS'].values
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 训练随机森林回归
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # 特征重要性
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 保存模型
        model_path = f"{self.output_dir}/models/solubility_model.pkl"
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, model_path)
        
        self.models['solubility'] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }
        
        print(f"✅ 训练完成!")
        print(f"   R² = {metrics['r2']:.3f}")
        print(f"   RMSE = {metrics['rmse']:.3f}")
        print(f"   MAE = {metrics['mae']:.3f}")
        print(f"   Top 5 特征: {[f[0] for f in top_features]}")
        
        return metrics
    
    def predict_solubility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测溶解度
        """
        print("\n🔮 预测溶解度...")
        
        if 'solubility' not in self.models:
            print("❌ 溶解度模型未训练")
            return df
        
        model_info = self.models['solubility']
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        X = df[feature_cols].values
        predictions = model.predict(X)
        
        df['LogS_predicted'] = predictions
        
        # 溶解度等级
        def get_solubility_class(logs):
            if logs > -2:
                return 'High'
            elif logs > -4:
                return 'Medium'
            else:
                return 'Low'
        
        df['solubility_class_predicted'] = df['LogS_predicted'].apply(get_solubility_class)
        
        # 人体可吸收性估算
        def get_absorption_estimate(logs):
            if logs > -1:
                return 'Excellent'
            elif logs > -2:
                return 'Good'
            elif logs > -3:
                return 'Moderate'
            elif logs > -4:
                return 'Poor'
            else:
                return 'Very Poor'
        
        df['absorption_estimate'] = df['LogS_predicted'].apply(get_absorption_estimate)
        
        print(f"✅ 溶解度预测完成")
        return df
    
    # ==================== 2. 代谢稳定性预测 ====================
    
    def calculate_metabolism_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算代谢稳定性相关特征
        """
        print("\n🔬 计算代谢稳定性特征...")
        
        def get_metabolism_descriptors(mol):
            if mol is None:
                return {}
            
            # CYP450代谢相关特征
            # 1. 可代谢位点计数
            num_metabolic_sites = self._count_metabolic_sites(mol)
            
            # 2. 特定官能团 (CYP底物特征)
            has_aromatic_methyl = self._has_aromatic_methyl(mol)
            has_benzylic_position = self._has_benzylic_position(mol)
            has_alkyl_chain = self._has_alkyl_chain(mol)
            has_N_dealkylation_site = self._has_N_dealkylation_site(mol)
            has_O_dealkylation_site = self._has_O_dealkylation_site(mol)
            
            # 3. 空间位阻 (影响代谢速率)
            num_branches = Lipinski.NumAliphaticCarbocycles(mol) + \
                          Lipinski.NumAliphaticHeterocycles(mol)
            
            # 4. 电子性质
            aromatic_density = Lipinski.NumAromaticRings(mol) / Descriptors.HeavyAtomCount(mol) if Descriptors.HeavyAtomCount(mol) > 0 else 0
            
            return {
                'NumMetabolicSites': num_metabolic_sites,
                'HasAromaticMethyl': has_aromatic_methyl,
                'HasBenzylicPosition': has_benzylic_position,
                'HasAlkylChain': has_alkyl_chain,
                'HasNDealkylation': has_N_dealkylation_site,
                'HasODealkylation': has_O_dealkylation_site,
                'BranchCount': num_branches,
                'AromaticDensity': aromatic_density,
                # 组合特征
                'MetabolicVulnerability': num_metabolic_sites / (num_branches + 1),
            }
        
        desc_df = df['mol'].apply(get_metabolism_descriptors).apply(pd.Series)
        df = pd.concat([df, desc_df], axis=1)
        
        print(f"✅ 计算完成: {len(desc_df.columns)} 个代谢特征")
        return df
    
    def _count_metabolic_sites(self, mol):
        """计算潜在代谢位点数"""
        count = 0
        # 芳香甲基
        pattern1 = Chem.MolFromSmarts('[c]C')
        if pattern1:
            count += len(mol.GetSubstructMatches(pattern1))
        # 苄位
        pattern2 = Chem.MolFromSmarts('[c]C[CH2]')
        if pattern2:
            count += len(mol.GetSubstructMatches(pattern2))
        # 叔胺
        pattern3 = Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]')
        if pattern3:
            count += len(mol.GetSubstructMatches(pattern3))
        return count
    
    def _has_aromatic_methyl(self, mol):
        """检测芳香甲基 (CYP常见底物)"""
        pattern = Chem.MolFromSmarts('[c]C')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_benzylic_position(self, mol):
        """检测苄位"""
        pattern = Chem.MolFromSmarts('[c]C[!H0]')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_alkyl_chain(self, mol):
        """检测烷基链 (>3个碳)"""
        pattern = Chem.MolFromSmarts('CCCC')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_N_dealkylation_site(self, mol):
        """检测N-去烷基化位点"""
        pattern = Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]C')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_O_dealkylation_site(self, mol):
        """检测O-去烷基化位点"""
        pattern = Chem.MolFromSmarts('[OX2]C')
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def create_metabolism_dataset(self) -> pd.DataFrame:
        """
        创建代谢稳定性示例数据集
        t1/2: 半衰期 (小时)
        """
        print("\n📦 创建代谢稳定性数据集...")
        
        # 高稳定性 (t1/2 > 8h)
        high_stability = [
            ('COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1', 12, 'high'),  # 奥氮平
            ('c1ccc2c(c1)c(c[nH]2)CCN', 10, 'high'),  # 色胺
        ]
        
        # 中等稳定性 (t1/2 3-8h)
        medium_stability = [
            ('CC(C)NCC(COc1ccccc1)O', 5, 'medium'),  # 普萘洛尔
            ('CN1CCC[C@H]1c2cccnc2', 4, 'medium'),  # 尼古丁
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 4, 'medium'),  # 布洛芬
        ]
        
        # 低稳定性 (t1/2 < 3h)
        low_stability = [
            ('CC(=O)Oc1ccccc1C(=O)O', 0.5, 'low'),  # 阿司匹林
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 2, 'low'),  # 咖啡因
        ]
        
        all_data = high_stability + medium_stability + low_stability
        
        df = pd.DataFrame(all_data, columns=['smiles', 't1/2_h', 'stability_class'])
        
        # 验证SMILES
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 创建数据集: {len(df)} 个分子")
        print(f"   高稳定性: {sum(df['stability_class'] == 'high')}")
        print(f"   中稳定性: {sum(df['stability_class'] == 'medium')}")
        print(f"   低稳定性: {sum(df['stability_class'] == 'low')}")
        
        return df
    
    def train_metabolism_model(self, df: pd.DataFrame) -> Dict:
        """
        训练代谢稳定性预测模型
        """
        print("\n🤖 训练代谢稳定性预测模型...")
        
        # 准备特征
        feature_cols = [col for col in df.columns if col not in 
                       ['smiles', 'mol', 't1/2_h', 'stability_class']]
        
        X = df[feature_cols].values
        y = df['t1/2_h'].values
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 训练随机森林回归
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # 保存模型
        model_path = f"{self.output_dir}/models/metabolism_model.pkl"
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, model_path)
        
        self.models['metabolism'] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }
        
        print(f"✅ 训练完成!")
        print(f"   R² = {metrics['r2']:.3f}")
        print(f"   RMSE = {metrics['rmse']:.3f} h")
        print(f"   MAE = {metrics['mae']:.3f} h")
        
        return metrics
    
    def predict_metabolism(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测代谢稳定性
        """
        print("\n🔮 预测代谢稳定性...")
        
        if 'metabolism' not in self.models:
            print("❌ 代谢模型未训练")
            return df
        
        model_info = self.models['metabolism']
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        X = df[feature_cols].values
        predictions = model.predict(X)
        
        df['t1/2_predicted'] = predictions
        
        # 稳定性等级
        def get_stability_class(t12):
            if t12 >= 8:
                return 'High'
            elif t12 >= 3:
                return 'Medium'
            else:
                return 'Low'
        
        df['stability_class_predicted'] = df['t1/2_predicted'].apply(get_stability_class)
        
        # 给药频率建议
        def get_dosing_frequency(t12):
            if t12 >= 12:
                return 'Once daily (QD)'
            elif t12 >= 8:
                return 'Twice daily (BID)'
            elif t12 >= 4:
                return 'Three times daily (TID)'
            else:
                return 'Four times daily (QID) or extended release'
        
        df['dosing_recommendation'] = df['t1/2_predicted'].apply(get_dosing_frequency)
        
        print(f"✅ 代谢稳定性预测完成")
        return df
    
    # ==================== 3. CYP450 抑制预测 ====================
    
    def create_cyp_dataset(self) -> pd.DataFrame:
        """
        创建 CYP450 抑制数据集
        """
        print("\n📦 创建 CYP450 数据集...")
        
        # CYP3A4 抑制剂
        cyp_inhibitors = [
            ('COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1', 1),  # 奥氮平
            ('c1ccc2c(c1)c(c[nH]2)CCN', 1),  # 色胺
        ]
        
        # CYP3A4 非抑制剂
        cyp_non_inhibitors = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 0),  # 布洛芬
            ('CC(=O)Oc1ccccc1C(=O)O', 0),  # 阿司匹林
            ('CC(C)NCC(COc1ccccc1)O', 0),  # 普萘洛尔
        ]
        
        all_data = cyp_inhibitors + cyp_non_inhibitors
        
        df = pd.DataFrame(all_data, columns=['smiles', 'cyp3a4_inhibitor'])
        
        # 验证SMILES
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 创建数据集: {len(df)} 个分子")
        print(f"   CYP抑制剂: {df['cyp3a4_inhibitor'].sum()}")
        print(f"   CYP非抑制剂: {len(df) - df['cyp3a4_inhibitor'].sum()}")
        
        return df
    
    def train_cyp_model(self, df: pd.DataFrame) -> Dict:
        """
        训练 CYP450 抑制预测模型
        """
        print("\n🤖 训练 CYP450 抑制预测模型...")
        
        # 使用溶解度和代谢特征的组合
        feature_cols = [col for col in df.columns if col not in 
                       ['smiles', 'mol', 'cyp3a4_inhibitor'] and 
                       df[col].dtype in ['int64', 'float64', 'bool']]
        
        if len(feature_cols) < 5:
            print("⚠️ 特征不足，使用基础分子描述符")
            # 添加基础描述符
            df['MW'] = df['mol'].apply(Descriptors.MolWt)
            df['LogP'] = df['mol'].apply(Crippen.MolLogP)
            df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
            feature_cols = ['MW', 'LogP', 'TPSA']
        
        X = df[feature_cols].values
        y = df['cyp3a4_inhibitor'].values
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练分类器
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # 保存模型
        model_path = f"{self.output_dir}/models/cyp_inhibition_model.pkl"
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, model_path)
        
        self.models['cyp'] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }
        
        print(f"✅ 训练完成!")
        print(f"   准确率: {metrics['accuracy']:.3f}")
        print(f"   F1分数: {metrics['f1']:.3f}")
        
        return metrics
    
    def predict_cyp_inhibition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测 CYP450 抑制
        """
        print("\n🔮 预测 CYP450 抑制...")
        
        if 'cyp' not in self.models:
            print("❌ CYP模型未训练")
            return df
        
        model_info = self.models['cyp']
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        # 确保所有特征都存在
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].values
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        df['cyp3a4_inhibitor_predicted'] = predictions
        df['cyp_inhibition_probability'] = probabilities
        
        # 药物相互作用风险
        def get_ddi_risk(prob):
            if prob > 0.7:
                return 'High DDI Risk'
            elif prob > 0.3:
                return 'Moderate DDI Risk'
            else:
                return 'Low DDI Risk'
        
        df['ddi_risk'] = df['cyp_inhibition_probability'].apply(get_ddi_risk)
        
        print(f"✅ CYP抑制预测完成")
        return df
    
    # ==================== 4. 生成综合ADMET报告 ====================
    
    def generate_admet_report(self, df: pd.DataFrame):
        """
        生成综合ADMET报告
        """
        print("\n📄 生成ADMET综合报告...")
        
        report = {
            'summary': {
                'total_molecules': len(df),
            },
            'solubility': {},
            'metabolism': {},
            'cyp': {}
        }
        
        # 溶解度统计
        if 'solubility_class_predicted' in df.columns:
            report['solubility'] = df['solubility_class_predicted'].value_counts().to_dict()
        
        # 代谢稳定性统计
        if 'stability_class_predicted' in df.columns:
            report['metabolism'] = df['stability_class_predicted'].value_counts().to_dict()
        
        # CYP统计
        if 'ddi_risk' in df.columns:
            report['cyp'] = df['ddi_risk'].value_counts().to_dict()
        
        # 保存报告
        report_path = f"{self.output_dir}/results/admet_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 保存详细结果
        result_path = f"{self.output_dir}/results/admet_predictions.csv"
        df.to_csv(result_path, index=False)
        
        print(f"✅ 报告保存: {report_path}")
        print(f"✅ 详细结果: {result_path}")
        
        return report


# ==================== 使用示例 ====================

def demo_admet_prediction():
    """
    演示ADMET预测完整流程
    """
    print("=" * 70)
    print("🧪 药学研究 AI 工作流 - 阶段二（续）：ADMET预测")
    print("=" * 70)
    
    # 初始化
    admet = ADMETPrediction(output_dir="./pharma_demo")
    
    # ==================== B. 溶解度预测 ====================
    print("\n" + "=" * 70)
    print("B. 溶解度预测模块")
    print("=" * 70)
    
    # 创建溶解度数据集
    df_sol = admet.create_solubility_dataset()
    
    # 计算特征
    df_sol = admet.calculate_solubility_features(df_sol)
    
    # 训练模型
    sol_metrics = admet.train_solubility_model(df_sol)
    
    # 预测
    df_sol = admet.predict_solubility(df_sol)
    
    print("\n溶解度预测结果:")
    print(df_sol[['smiles', 'LogS_predicted', 'solubility_class_predicted', 'absorption_estimate']].to_string())
    
    # ==================== C. 代谢稳定性预测 ====================
    print("\n" + "=" * 70)
    print("C. 代谢稳定性预测模块")
    print("=" * 70)
    
    # 创建代谢数据集
    df_met = admet.create_metabolism_dataset()
    
    # 计算特征
    df_met = admet.calculate_metabolism_features(df_met)
    
    # 训练模型
    met_metrics = admet.train_metabolism_model(df_met)
    
    # 预测
    df_met = admet.predict_metabolism(df_met)
    
    print("\n代谢稳定性预测结果:")
    print(df_met[['smiles', 't1/2_predicted', 'stability_class_predicted', 'dosing_recommendation']].to_string())
    
    # CYP450 抑制预测
    print("\n" + "-" * 70)
    print("CYP450 抑制预测")
    print("-" * 70)
    
    df_cyp = admet.create_cyp_dataset()
    df_cyp = admet.calculate_solubility_features(df_cyp)
    df_cyp = admet.calculate_metabolism_features(df_cyp)
    
    cyp_metrics = admet.train_cyp_model(df_cyp)
    df_cyp = admet.predict_cyp_inhibition(df_cyp)
    
    print("\nCYP抑制预测结果:")
    print(df_cyp[['smiles', 'cyp_inhibition_probability', 'ddi_risk']].to_string())
    
    # 生成报告
    print("\n" + "=" * 70)
    print("生成ADMET综合报告")
    print("=" * 70)
    
    # 合并所有预测结果
    all_smiles = list(set(df_sol['smiles'].tolist() + df_met['smiles'].tolist()))
    
    report = admet.generate_admet_report(df_sol)
    
    print("\n" + "=" * 70)
    print("✅ ADMET预测模块完成!")
    print("=" * 70)
    print("\n📊 模型性能总结:")
    print(f"   溶解度预测 R²: {sol_metrics['r2']:.3f}")
    print(f"   代谢稳定性 R²: {met_metrics['r2']:.3f}")
    print(f"   CYP抑制 F1: {cyp_metrics['f1']:.3f}")
    print("=" * 70)
    
    return admet, df_sol, df_met, df_cyp


if __name__ == "__main__":
    admet, df_sol, df_met, df_cyp = demo_admet_prediction()
