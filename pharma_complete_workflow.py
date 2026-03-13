#!/usr/bin/env python3
"""
药学研究 AI 工作流 - 完整整合版 v1.0
PharmaAI Complete Workflow

整合模块：
1. 数据增强与验证 (DataEnhancement)
2. 分子性质预测 (PropertyPrediction)
3. 毒性预测 (ToxicityPrediction)
4. ADMET预测 (ADMETPrediction)
5. 虚拟筛选 (VirtualScreening)
6. 可视化与报告 (Visualization & Reporting)

作者: PharmaAI Workflow
版本: 1.0
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd

# 化学信息学
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Draw, AllChem, DataStructs,
    PandasTools, Crippen, Lipinski, rdMolDescriptors
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# 机器学习
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

warnings.filterwarnings('ignore')


# ==================== 数据类定义 ====================

@dataclass
class Molecule:
    """分子数据类"""
    smiles: str
    mol: Optional[Chem.Mol] = None
    properties: Dict = None
    predictions: Dict = None
    
    def __post_init__(self):
        if self.mol is None and self.smiles:
            self.mol = Chem.MolFromSmiles(self.smiles)
        if self.properties is None:
            self.properties = {}
        if self.predictions is None:
            self.predictions = {}


@dataclass
class WorkflowConfig:
    """工作流配置"""
    output_dir: str = "./pharma_workflow"
    enable_data_cleaning: bool = True
    enable_lipinski_filter: bool = True
    enable_toxicity: bool = True
    enable_solubility: bool = True
    enable_metabolism: bool = True
    enable_cyp: bool = True
    virtual_screening_top_n: int = 100
    
    def to_dict(self):
        return asdict(self)


# ==================== 主工作流类 ====================

class PharmaAICompleteWorkflow:
    """
    药学研究 AI 完整工作流
    
    使用示例:
        workflow = PharmaAICompleteWorkflow()
        results = workflow.run_complete_pipeline("your_data.csv")
    """
    
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.output_dir = self.config.output_dir
        self.models = {}
        self.datasets = {}
        
        # 创建目录结构
        self._create_directories()
        
        # 初始化日志
        self.log_file = f"{self.output_dir}/workflow.log"
        self._log(f"PharmaAI Workflow v1.0 初始化")
        self._log(f"配置: {json.dumps(self.config.to_dict(), indent=2)}")
        
        print(f"✅ PharmaAI 完整工作流初始化完成")
        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")
    
    def _create_directories(self):
        """创建工作目录"""
        dirs = [
            f"{self.output_dir}",
            f"{self.output_dir}/data",
            f"{self.output_dir}/models",
            f"{self.output_dir}/results",
            f"{self.output_dir}/visualizations",
            f"{self.output_dir}/reports"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    # ==================== 1. 数据加载与预处理 ====================
    
    def load_data(self, file_path: str, smiles_col: str = 'smiles',
                  activity_col: Optional[str] = None) -> pd.DataFrame:
        """
        加载分子数据
        
        支持格式: CSV, SDF, Excel
        """
        self._log(f"加载数据: {file_path}")
        print(f"\n📥 加载数据: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.sdf'):
            df = PandasTools.LoadSDF(file_path, smilesName=smiles_col)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("不支持的文件格式。请使用 CSV, SDF, 或 Excel")
        
        # 验证SMILES
        df['mol'] = df[smiles_col].apply(
            lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None
        )
        valid_count = df['mol'].notna().sum()
        
        print(f"   总记录: {len(df)}")
        print(f"   有效分子: {valid_count}")
        
        self._log(f"数据加载完成: {valid_count}/{len(df)} 有效")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗: 去重、标准化、过滤
        """
        if not self.config.enable_data_cleaning:
            return df
        
        self._log("开始数据清洗")
        print(f"\n🧹 数据清洗...")
        
        initial_count = len(df)
        
        # 1. 去除无效分子
        df = df[df['mol'].notna()].copy()
        
        # 2. 标准化SMILES
        def canonicalize(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol, canonical=True) if mol else None
            except:
                return None
        
        df['smiles_canonical'] = df['smiles'].apply(canonicalize)
        df = df[df['smiles_canonical'].notna()]
        
        # 3. 去重
        df = df.drop_duplicates(subset=['smiles_canonical'], keep='first')
        
        # 4. 过滤异常分子
        df['MW'] = df['mol'].apply(Descriptors.MolWt)
        df = df[(df['MW'] >= 50) & (df['MW'] <= 1000)]
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"   清洗前: {initial_count}")
        print(f"   清洗后: {final_count}")
        print(f"   移除: {removed} ({removed/initial_count*100:.1f}%)")
        
        self._log(f"数据清洗完成: {final_count} 保留, {removed} 移除")
        
        return df
    
    def apply_lipinski_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用 Lipinski 五规则筛选
        """
        if not self.config.enable_lipinski_filter:
            return df
        
        print(f"\n💊 Lipinski 五规则筛选...")
        
        # 计算描述符
        df['LogP'] = df['mol'].apply(Crippen.MolLogP)
        df['HBD'] = df['mol'].apply(Lipinski.NumHDonors)
        df['HBA'] = df['mol'].apply(Lipinski.NumHAcceptors)
        
        # 检查规则
        def check_lipinski(row):
            violations = 0
            if row['MW'] > 500: violations += 1
            if row['LogP'] > 5: violations += 1
            if row['HBD'] > 5: violations += 1
            if row['HBA'] > 10: violations += 1
            return violations <= 1
        
        df['lipinski_pass'] = df.apply(check_lipinski, axis=1)
        filtered = df[df['lipinski_pass']].copy()
        
        print(f"   通过: {len(filtered)}/{len(df)} ({len(filtered)/len(df)*100:.1f}%)")
        
        return filtered
    
    # ==================== 2. 特征工程 ====================
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有分子特征
        """
        print(f"\n🔬 计算分子特征...")
        
        # 基础描述符
        df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
        df['RotatableBonds'] = df['mol'].apply(Lipinski.NumRotatableBonds)
        df['AromaticRings'] = df['mol'].apply(Lipinski.NumAromaticRings)
        df['HeavyAtoms'] = df['mol'].apply(lambda m: m.GetNumHeavyAtoms())
        
        # 毒性相关特征
        df['NumNitrogens'] = df['mol'].apply(
            lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)
        )
        df['NumHalogens'] = df['mol'].apply(
            lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
        )
        
        # 溶解度相关
        df['MolRefractivity'] = df['mol'].apply(Crippen.MolMR)
        
        # Morgan指纹 (用于机器学习)
        print("   计算 Morgan 指纹...")
        fingerprints = []
        for mol in df['mol']:
            if mol:
                fp = GetMorganFingerprintAsBitVect(mol, 2, 2048)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(2048))
        df['fingerprint'] = fingerprints
        
        print(f"✅ 特征计算完成")
        return df
    
    # ==================== 3. 模型训练 ====================
    
    def train_models(self, df: pd.DataFrame, 
                     target_col: Optional[str] = None) -> Dict:
        """
        训练所有预测模型
        """
        self._log("开始模型训练")
        print(f"\n🤖 训练预测模型...")
        
        results = {}
        
        # 如果提供了活性数据，训练活性预测模型
        if target_col and target_col in df.columns:
            print(f"\n   训练活性预测模型...")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            X = np.stack(df['fingerprint'].values)
            y = df[target_col].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            self.models['activity'] = model
            results['activity'] = metrics
            
            print(f"   R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")
            
            # 保存模型
            joblib.dump(model, f"{self.output_dir}/models/activity_model.pkl")
        
        # 毒性预测模型 (简化版，基于规则)
        if self.config.enable_toxicity:
            print(f"\n   初始化毒性预测模型...")
            # 这里可以加载预训练的毒性模型
            results['toxicity'] = 'rule-based'
        
        # ADMET模型
        if self.config.enable_solubility:
            print(f"   初始化溶解度预测模型...")
            results['solubility'] = 'rule-based'
        
        self._log(f"模型训练完成: {list(results.keys())}")
        return results
    
    # ==================== 4. 预测与筛选 ====================
    
    def predict_all_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测所有分子性质
        """
        print(f"\n🔮 预测分子性质...")
        
        # 活性预测
        if 'activity' in self.models:
            print("   预测活性...")
            X = np.stack(df['fingerprint'].values)
            df['activity_predicted'] = self.models['activity'].predict(X)
        
        # 毒性预测 (基于规则)
        if self.config.enable_toxicity:
            print("   预测毒性...")
            df['toxicity_risk'] = self._predict_toxicity_risk(df)
        
        # 溶解度预测
        if self.config.enable_solubility:
            print("   预测溶解度...")
            df['solubility_class'] = self._predict_solubility_class(df)
        
        # 综合评分
        df['overall_score'] = self._calculate_overall_score(df)
        
        print(f"✅ 预测完成")
        return df
    
    def _predict_toxicity_risk(self, df: pd.DataFrame) -> pd.Series:
        """基于规则的毒性风险评估"""
        risk_scores = np.zeros(len(df))
        
        # hERG风险: 碱性胺 + 高LogP
        risk_scores += ((df['NumNitrogens'] > 0) & (df['LogP'] > 2)).astype(int) * 0.3
        
        # 肝毒性风险: 卤代芳烃
        risk_scores += (df['NumHalogens'] > 0).astype(int) * 0.2
        
        # 分类
        def get_risk(score):
            if score >= 0.3:
                return 'High'
            elif score >= 0.1:
                return 'Medium'
            return 'Low'
        
        return pd.Series(risk_scores).apply(get_risk)
    
    def _predict_solubility_class(self, df: pd.DataFrame) -> pd.Series:
        """基于规则的溶解度分类"""
        # 简化规则: TPSA/MW 比值越高，溶解度越好
        tpsa_ratio = df['TPSA'] / df['MW']
        
        def get_class(ratio):
            if ratio > 0.25:
                return 'High'
            elif ratio > 0.15:
                return 'Medium'
            return 'Low'
        
        return tpsa_ratio.apply(get_class)
    
    def _calculate_overall_score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合评分"""
        scores = np.zeros(len(df))
        
        # 活性分数 (如果有)
        if 'activity_predicted' in df.columns:
            scores += df['activity_predicted'] * 0.4
        
        # 毒性惩罚
        toxicity_penalty = df['toxicity_risk'].map({'Low': 0, 'Medium': -0.2, 'High': -0.5})
        scores += toxicity_penalty.fillna(0)
        
        # 溶解度奖励
        solubility_bonus = df['solubility_class'].map({'Low': -0.1, 'Medium': 0, 'High': 0.1})
        scores += solubility_bonus.fillna(0)
        
        # Lipinski奖励
        if 'lipinski_pass' in df.columns:
            scores += df['lipinski_pass'].astype(int) * 0.1
        
        return scores
    
    def virtual_screening(self, df: pd.DataFrame, 
                         top_n: Optional[int] = None) -> pd.DataFrame:
        """
        虚拟筛选: 选择最佳候选药物
        """
        if top_n is None:
            top_n = self.config.virtual_screening_top_n
        
        print(f"\n🔍 虚拟筛选 (Top {top_n})...")
        
        # 按综合评分排序
        top_candidates = df.nlargest(top_n, 'overall_score')
        
        # 保存结果
        output_file = f"{self.output_dir}/results/top_candidates.csv"
        top_candidates.to_csv(output_file, index=False)
        
        print(f"✅ 筛选完成: {len(top_candidates)} 个候选")
        print(f"   保存: {output_file}")
        
        return top_candidates
    
    # ==================== 5. 可视化与报告 ====================
    
    def visualize_candidates(self, df: pd.DataFrame, 
                            filename: str = "candidates.png"):
        """
        可视化候选分子
        """
        print(f"\n🎨 可视化分子...")
        
        # 选择前12个
        mols = df['mol'].head(12).tolist()
        legends = []
        
        if 'overall_score' in df.columns:
            legends = [f"Score: {s:.2f}" for s in df['overall_score'].head(12)]
        
        if mols:
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=4,
                subImgSize=(300, 300),
                legends=legends
            )
            
            output_path = f"{self.output_dir}/visualizations/{filename}"
            img.save(output_path)
            print(f"✅ 图片保存: {output_path}")
    
    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                     model_metrics: Dict) -> Dict:
        """
        生成综合报告
        """
        print(f"\n📄 生成综合报告...")
        
        report = {
            'workflow_info': {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict()
            },
            'data_summary': {
                'total_molecules': len(df),
                'lipinski_pass_rate': float(df['lipinski_pass'].mean()) if 'lipinski_pass' in df.columns else None,
            },
            'model_performance': model_metrics,
            'property_distribution': {},
            'top_candidates': []
        }
        
        # 属性分布
        if 'toxicity_risk' in df.columns:
            report['property_distribution']['toxicity'] = df['toxicity_risk'].value_counts().to_dict()
        
        if 'solubility_class' in df.columns:
            report['property_distribution']['solubility'] = df['solubility_class'].value_counts().to_dict()
        
        # Top候选
        if 'overall_score' in df.columns:
            top_df = df.nlargest(5, 'overall_score')
            for _, row in top_df.iterrows():
                candidate = {
                    'smiles': row['smiles'],
                    'overall_score': float(row['overall_score'])
                }
                if 'activity_predicted' in row:
                    candidate['activity'] = float(row['activity_predicted'])
                if 'toxicity_risk' in row:
                    candidate['toxicity'] = row['toxicity_risk']
                report['top_candidates'].append(candidate)
        
        # 保存报告
        report_path = f"{self.output_dir}/reports/comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 保存完整数据
        data_path = f"{self.output_dir}/results/complete_predictions.csv"
        df.to_csv(data_path, index=False)
        
        print(f"✅ 报告保存: {report_path}")
        print(f"✅ 数据保存: {data_path}")
        
        return report
    
    # ==================== 6. 完整流程 ====================
    
    def run_complete_pipeline(self, input_file: str,
                             smiles_col: str = 'smiles',
                             activity_col: Optional[str] = None) -> Dict:
        """
        运行完整工作流
        
        这是主入口函数，执行从数据加载到报告生成的完整流程。
        
        参数:
            input_file: 输入数据文件路径 (CSV/SDF/Excel)
            smiles_col: SMILES列名
            activity_col: 生物活性列名 (可选)
        
        返回:
            包含所有结果的字典
        """
        print("=" * 70)
        print("🧪 PharmaAI 完整工作流")
        print("=" * 70)
        
        self._log("开始完整流程")
        
        # 步骤1: 加载数据
        df = self.load_data(input_file, smiles_col, activity_col)
        
        # 步骤2: 数据清洗
        df = self.clean_data(df)
        
        # 步骤3: Lipinski筛选
        df = self.apply_lipinski_filter(df)
        
        # 步骤4: 特征工程
        df = self.calculate_all_features(df)
        
        # 步骤5: 模型训练
        model_metrics = self.train_models(df, activity_col)
        
        # 步骤6: 性质预测
        df = self.predict_all_properties(df)
        
        # 步骤7: 虚拟筛选
        top_candidates = self.virtual_screening(df)
        
        # 步骤8: 可视化
        self.visualize_candidates(top_candidates)
        
        # 步骤9: 生成报告
        report = self.generate_comprehensive_report(df, model_metrics)
        
        # 完成
        print("\n" + "=" * 70)
        print("✅ 工作流完成!")
        print("=" * 70)
        print(f"\n📊 结果摘要:")
        print(f"   总分子数: {len(df)}")
        print(f"   Top候选: {len(top_candidates)}")
        print(f"   输出目录: {os.path.abspath(self.output_dir)}")
        
        self._log("工作流完成")
        
        return {
            'dataset': df,
            'top_candidates': top_candidates,
            'report': report,
            'model_metrics': model_metrics
        }
    
    def quick_predict(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        快速预测: 对新分子进行ADMET预测
        
        使用示例:
            results = workflow.quick_predict(['CCO', 'CC(C)O'])
        """
        print(f"\n⚡ 快速预测 {len(smiles_list)} 个分子...")
        
        # 创建DataFrame
        df = pd.DataFrame({'smiles': smiles_list})
        df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
        df = df[df['mol'].notna()]
        
        # 计算特征
        df = self.calculate_all_features(df)
        
        # 基础描述符
        df['MW'] = df['mol'].apply(Descriptors.MolWt)
        df['LogP'] = df['mol'].apply(Crippen.MolLogP)
        df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
        
        # 预测
        df['toxicity_risk'] = self._predict_toxicity_risk(df)
        df['solubility_class'] = self._predict_solubility_class(df)
        
        print(f"✅ 预测完成")
        
        return df[['smiles', 'MW', 'LogP', 'TPSA', 'toxicity_risk', 'solubility_class']]


# ==================== 使用示例 ====================

def demo_complete_workflow():
    """
    演示完整工作流
    """
    # 创建工作流实例
    config = WorkflowConfig(
        output_dir="./pharma_complete",
        enable_data_cleaning=True,
        enable_lipinski_filter=True,
        enable_toxicity=True,
        enable_solubility=True,
        virtual_screening_top_n=5
    )
    
    workflow = PharmaAICompleteWorkflow(config)
    
    # 创建示例数据
    print("\n" + "=" * 70)
    print("创建示例数据集...")
    print("=" * 70)
    
    sample_data = {
        'smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # 布洛芬
            'CC(=O)Oc1ccccc1C(=O)O',  # 阿司匹林
            'CC(C)NCC(COc1ccccc1)O',  # 普萘洛尔
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
            'COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1',  # 奥氮平
            'c1ccc(cc1)C(=O)O',  # 苯甲酸
            'c1ccccc1',  # 苯
            'CCO',  # 乙醇
        ],
        'activity': [0.85, 0.82, 0.91, 0.75, 0.88, 0.30, 0.05, 0.10]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = "./pharma_complete/data/sample_input.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"✅ 示例数据保存: {sample_file}")
    
    # 运行完整流程
    results = workflow.run_complete_pipeline(
        input_file=sample_file,
        smiles_col='smiles',
        activity_col='activity'
    )
    
    # 快速预测示例
    print("\n" + "=" * 70)
    print("快速预测新分子示例")
    print("=" * 70)
    
    new_molecules = [
        'CC(C)Cc1ccc(cc1)C(=O)O',  # 布洛芬
        'c1ccc2c(c1)c(c[nH]2)CCN',  # 色胺
    ]
    
    quick_results = workflow.quick_predict(new_molecules)
    print("\n快速预测结果:")
    print(quick_results.to_string())
    
    return workflow, results


if __name__ == "__main__":
    workflow, results = demo_complete_workflow()
