#!/usr/bin/env python3
"""
PharmaAI 完整模型训练与集成
训练所有毒性模型并集成到工作流
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdFingerprintGenerator

# 初始化Morgan指纹生成器 (避免弃用警告)
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)



class CompleteModelTrainer:
    """完整模型训练器"""
    
    def __init__(self, output_dir: str = "./pharma_models"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        self.models = {}
        
        print(f"✅ 完整模型训练器初始化")
    
    def create_hepatotoxicity_dataset(self) -> pd.DataFrame:
        """创建肝毒性数据集"""
        print(f"\n📦 创建肝毒性数据集...")
        
        # 肝毒性阳性 (已知肝毒性药物)
        hepatotoxic = [
            ('CC(=O)Nc1ccc(O)cc1', 1),  # 对乙酰氨基酚
            ('CC(C)NCC(COc1ccccc1)O', 1),  # 普萘洛尔
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 1),  # 咖啡因
            ('COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1', 1),  # 奥氮平
            ('c1ccc2c(c1)c(c[nH]2)CCN', 1),  # 色胺
        ]
        
        # 肝毒性阴性 (安全药物)
        safe = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 0),  # 布洛芬
            ('CC(=O)Oc1ccccc1C(=O)O', 0),  # 阿司匹林
            ('c1ccc(cc1)C(=O)O', 0),  # 苯甲酸
            ('CCO', 0),  # 乙醇
            ('c1ccccc1', 0),  # 苯
        ]
        
        all_data = hepatotoxic + safe
        df = pd.DataFrame(all_data, columns=['smiles', 'hepatotoxic'])
        
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 肝毒性数据: {len(df)} (阳性: {sum(df['hepatotoxic'])}, 阴性: {sum(1-df['hepatotoxic'])})")
        return df
    
    def create_ames_dataset(self) -> pd.DataFrame:
        """创建Ames致突变性数据集"""
        print(f"\n📦 创建Ames数据集...")
        
        # Ames阳性 (致突变)
        ames_positive = [
            ('O=[N+]([O-])c1ccccc1', 1),  # 硝基苯
            ('Nc1ccccc1', 1),  # 苯胺
            ('CN(C)c1ccc(N=Nc2ccc(C)cc2)cc1', 1),  # 甲基黄
            ('c1ccc2c(c1)ccc3c2ccc4c3cccc4', 1),  # 苯并芘
        ]
        
        # Ames阴性
        ames_negative = [
            ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 0),  # 布洛芬
            ('CC(=O)Oc1ccccc1C(=O)O', 0),  # 阿司匹林
            ('CCO', 0),  # 乙醇
            ('c1ccc(cc1)C(=O)O', 0),  # 苯甲酸
            ('c1ccccc1O', 0),  # 苯酚
        ]
        
        all_data = ames_positive + ames_negative
        df = pd.DataFrame(all_data, columns=['smiles', 'ames_mutagen'])
        
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ Ames数据: {len(df)} (阳性: {sum(df['ames_mutagen'])}, 阴性: {sum(1-df['ames_mutagen'])})")
        return df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算分子特征"""
        df['MW'] = df['mol'].apply(Descriptors.MolWt)
        df['LogP'] = df['mol'].apply(Crippen.MolLogP)
        df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
        df['HBD'] = df['mol'].apply(Lipinski.NumHDonors)
        df['HBA'] = df['mol'].apply(Lipinski.NumHAcceptors)
        df['RotatableBonds'] = df['mol'].apply(Lipinski.NumRotatableBonds)
        df['AromaticRings'] = df['mol'].apply(Lipinski.NumAromaticRings)
        df['HeavyAtoms'] = df['mol'].apply(lambda m: m.GetNumHeavyAtoms())
        df['NumNitrogens'] = df['mol'].apply(lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7))
        df['NumHalogens'] = df['mol'].apply(lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]))
        
        # Morgan指纹
        fingerprints = []
        for mol in df['mol']:
            if mol:
                fp = _morgan_generator.GetFingerprint(mol)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(2048))
        df['fingerprint'] = fingerprints
        
        return df
    
    def train_model(self, df: pd.DataFrame, target_col: str, model_name: str) -> dict:
        """训练模型"""
        print(f"\n🤖 训练{model_name}模型...")
        
        desc_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 
                    'AromaticRings', 'HeavyAtoms', 'NumNitrogens', 'NumHalogens']
        
        X_desc = df[desc_cols].values
        X_fp = np.stack(df['fingerprint'].values)
        X = np.hstack([X_desc, X_fp])
        y = df[target_col].values
        
        print(f"   样本: {len(df)} (正例: {sum(y)}, 负例: {sum(1-y)})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
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
        
        # 保存模型
        model_data = {
            'model': model,
            'desc_cols': desc_cols,
            'metrics': metrics,
            'model_name': model_name
        }
        
        model_path = f"{self.output_dir}/models/{model_name}_model.pkl"
        joblib.dump(model_data, model_path)
        self.models[model_name] = model_data
        
        print(f"✅ {model_name}模型完成!")
        print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"   F1: {metrics['f1']:.3f}")
        
        return metrics
    
    def train_all_models(self):
        """训练所有模型"""
        print("=" * 70)
        print("🧪 训练所有毒性预测模型")
        print("=" * 70)
        
        # 1. hERG模型 (已训练，重新加载)
        print("\n1. hERG模型 (已存在)")
        try:
            herg_model = joblib.load(f"{self.output_dir}/models/herg_model.pkl")
            self.models['herg'] = herg_model
            print("   已加载现有模型")
        except:
            print("   未找到现有模型")
        
        # 2. 肝毒性模型
        print("\n2. 肝毒性模型")
        df_hep = self.create_hepatotoxicity_dataset()
        df_hep = self.calculate_features(df_hep)
        metrics_hep = self.train_model(df_hep, 'hepatotoxic', 'hepatotoxicity')
        
        # 3. Ames模型
        print("\n3. Ames致突变性模型")
        df_ames = self.create_ames_dataset()
        df_ames = self.calculate_features(df_ames)
        metrics_ames = self.train_model(df_ames, 'ames_mutagen', 'ames')
        
        # 生成综合报告
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("✅ 所有模型训练完成!")
        print("=" * 70)
        
        return self.models
    
    def generate_summary_report(self):
        """生成综合报告"""
        report = "# PharmaAI 模型综合报告\n\n"
        report += "## 已训练模型\n\n"
        
        for name, model_data in self.models.items():
            metrics = model_data.get('metrics', {})
            report += f"### {name.upper()} 模型\n"
            report += f"- ROC-AUC: {metrics.get('roc_auc', 0):.3f}\n"
            report += f"- F1 Score: {metrics.get('f1', 0):.3f}\n"
            report += f"- 准确率: {metrics.get('accuracy', 0):.3f}\n\n"
        
        report_path = f"{self.output_dir}/models_summary.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 综合报告: {report_path}")


if __name__ == "__main__":
    trainer = CompleteModelTrainer()
    trainer.train_all_models()
