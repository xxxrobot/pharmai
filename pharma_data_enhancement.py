#!/usr/bin/env python3
"""
药学研究 AI 工作流 - 阶段一：数据增强与验证
包含：
1. 公共数据库数据获取 (ChEMBL, PubChem)
2. 数据预处理与清洗
3. 数据验证与质量评估
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter

# 化学信息学
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
from rdkit.Chem import PandasTools, Crippen, Lipinski
from rdkit.Chem import rdFingerprintGenerator

# 数据获取
import urllib.request
import urllib.parse

# 初始化Morgan指纹生成器 (避免弃用警告)
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


warnings.filterwarnings('ignore')


class DataEnhancement:
    """数据增强与验证模块"""
    
    def __init__(self, output_dir: str = "./pharma_demo"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        print(f"✅ 数据增强模块初始化")
    
    # ==================== 1. 公共数据库数据获取 ====================
    
    def fetch_chembl_data(self, target_id: str = None, 
                          max_results: int = 1000) -> pd.DataFrame:
        """
        从 ChEMBL 获取生物活性数据
        
        target_id: ChEMBL 靶点ID (如 'CHEMBL204' for HERG)
        """
        print(f"\n📥 从 ChEMBL 获取数据...")
        print(f"   靶点: {target_id or '所有靶点'}")
        
        try:
            # 使用 ChEMBL REST API
            base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            
            if target_id:
                url = f"{base_url}?target_chembl_id={target_id}&limit={max_results}"
            else:
                url = f"{base_url}?limit={max_results}"
            
            print(f"   请求URL: {url[:80]}...")
            
            # 发送请求
            req = urllib.request.Request(
                url,
                headers={'Accept': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # 解析数据
            activities = data.get('activities', [])
            
            records = []
            for activity in activities:
                record = {
                    'chembl_id': activity.get('molecule_chembl_id', ''),
                    'smiles': activity.get('canonical_smiles', ''),
                    'target_id': activity.get('target_chembl_id', ''),
                    'target_name': activity.get('target_pref_name', ''),
                    'activity_type': activity.get('standard_type', ''),
                    'activity_value': activity.get('standard_value', None),
                    'activity_unit': activity.get('standard_units', ''),
                    'pchembl_value': activity.get('pchembl_value', None),
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # 过滤有效SMILES
            df = df[df['smiles'].notna() & (df['smiles'] != '')]
            df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
            df = df[df['mol'].notna()]
            
            print(f"✅ 成功获取 {len(df)} 条活性数据")
            
            # 保存
            output_file = f"{self.output_dir}/data/chembl_{target_id or 'all'}.csv"
            df.to_csv(output_file, index=False)
            print(f"   数据保存: {output_file}")
            
            return df
            
        except Exception as e:
            print(f"❌ 获取失败: {e}")
            return pd.DataFrame()
    
    def fetch_pubchem_compounds(self, 
                                compound_list: List[str] = None,
                                property_names: List[str] = None) -> pd.DataFrame:
        """
        从 PubChem 获取化合物数据
        
        compound_list: CID列表或SMILES列表
        property_names: 需要获取的属性
        """
        print(f"\n📥 从 PubChem 获取数据...")
        
        if property_names is None:
            property_names = [
                'MolecularFormula', 'MolecularWeight', 'CanonicalSMILES',
                'IsomericSMILES', 'InChI', 'InChIKey',
                'IUPACName', 'XLogP', 'ExactMass',
                'MonoisotopicMass', 'TPSA', 'Complexity',
                'Charge', 'HBondDonorCount', 'HBondAcceptorCount',
                'RotatableBondCount', 'HeavyAtomCount', 'AtomStereoCount'
            ]
        
        # 示例：获取阿司匹林数据
        if compound_list is None:
            compound_list = ['2244']  # 阿司匹林 CID
        
        try:
            cids = ','.join(map(str, compound_list))
            properties = ','.join(property_names)
            
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/{properties}/JSON"
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            compounds = data.get('PropertyTable', {}).get('Properties', [])
            
            records = []
            for compound in compounds:
                records.append(compound)
            
            df = pd.DataFrame(records)
            
            print(f"✅ 成功获取 {len(df)} 个化合物")
            
            # 保存
            output_file = f"{self.output_dir}/data/pubchem_compounds.csv"
            df.to_csv(output_file, index=False)
            
            return df
            
        except Exception as e:
            print(f"❌ 获取失败: {e}")
            return pd.DataFrame()
    
    def create_sample_dataset(self, dataset_type: str = 'balanced') -> pd.DataFrame:
        """
        创建示例数据集（用于演示和测试）
        
        dataset_type: 'balanced' (平衡), 'imbalanced' (不平衡), 'large' (大规模)
        """
        print(f"\n📦 创建示例数据集 ({dataset_type})...")
        
        if dataset_type == 'balanced':
            # 平衡数据集：活性分子和非活性分子
            data = {
                'smiles': [
                    # 高活性分子 (药物)
                    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # 布洛芬
                    'CC(=O)Oc1ccccc1C(=O)O',  # 阿司匹林
                    'CC(C)NCC(COc1ccccc1)O',  # 普萘洛尔
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
                    'COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1',  # 奥氮平
                    'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # 布洛芬类似物
                    'CC(=O)Oc1ccc(C)cc1C(=O)O',  # 阿司匹林类似物
                    'CC(C)NCC(O)COc1ccccc1',  # 普萘洛尔类似物
                    
                    # 低活性分子
                    'c1ccc(cc1)C(=O)O',  # 苯甲酸
                    'c1ccccc1',  # 苯
                    'CCO',  # 乙醇
                    'CCCC',  # 丁烷
                    'CC(C)C',  # 异丁烷
                    'c1ccc(cc1)O',  # 苯酚
                    'CC(=O)O',  # 乙酸
                    'CC(C)O',  # 异丙醇
                ],
                'activity': [0.85, 0.82, 0.91, 0.75, 0.88, 0.83, 0.80, 0.89,
                           0.15, 0.05, 0.10, 0.02, 0.03, 0.12, 0.08, 0.11],
                'class': ['active']*8 + ['inactive']*8
            }
            
        elif dataset_type == 'large':
            # 大规模数据集（模拟）
            np.random.seed(42)
            n_samples = 1000
            
            # 生成随机SMILES（简化版，实际应使用有效SMILES）
            fragments = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's', 
                        'CC', 'CN', 'CO', 'CCO', 'c1ccccc1', 'CC(C)']
            
            smiles_list = []
            for _ in range(n_samples):
                # 随机组合片段
                n_frag = np.random.randint(2, 6)
                smi = ''.join(np.random.choice(fragments, n_frag))
                smiles_list.append(smi)
            
            # 模拟活性（基于分子量相关）
            activities = np.random.beta(2, 5, n_samples)  # 偏低的活性分布
            
            data = {
                'smiles': smiles_list,
                'activity': activities,
                'class': ['active' if a > 0.5 else 'inactive' for a in activities]
            }
            
        else:  # imbalanced
            # 不平衡数据集（模拟真实场景）
            np.random.seed(42)
            n_active = 50
            n_inactive = 450
            
            data = {
                'smiles': ['C'*i for i in range(10, 510)],  # 占位符
                'activity': list(np.random.uniform(0.6, 1.0, n_active)) + 
                           list(np.random.uniform(0, 0.3, n_inactive)),
                'class': ['active']*n_active + ['inactive']*n_inactive
            }
        
        df = pd.DataFrame(data)
        
        # 验证SMILES
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        df = df[df['mol'].notna()]
        
        print(f"✅ 创建数据集: {len(df)} 个分子")
        print(f"   活性: {sum(df['class'] == 'active')}")
        print(f"   非活性: {sum(df['class'] == 'inactive')}")
        
        # 保存
        output_file = f"{self.output_dir}/data/sample_{dataset_type}.csv"
        df.to_csv(output_file, index=False)
        print(f"   保存: {output_file}")
        
        return df
    
    # ==================== 2. 数据预处理与清洗 ====================
    
    def clean_dataset(self, df: pd.DataFrame, 
                      smiles_col: str = 'smiles') -> pd.DataFrame:
        """
        数据清洗：去重、标准化、处理无效数据
        """
        print(f"\n🧹 数据清洗...")
        print(f"   原始数据: {len(df)} 条")
        
        initial_count = len(df)
        
        # 1. 去除重复SMILES
        df = df.drop_duplicates(subset=[smiles_col], keep='first')
        print(f"   去重后: {len(df)} 条 (移除 {initial_count - len(df)} 条重复)")
        
        # 2. 标准化SMILES (Canonical SMILES)
        def canonicalize_smiles(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
                return None
            except:
                return None
        
        df['smiles_canonical'] = df[smiles_col].apply(canonicalize_smiles)
        df = df[df['smiles_canonical'].notna()]
        
        # 再次去重（canonical后可能产生新的重复）
        df = df.drop_duplicates(subset=['smiles_canonical'], keep='first')
        
        # 3. 去除无效分子
        df['mol'] = df['smiles_canonical'].apply(Chem.MolFromSmiles)
        df = df[df['mol'].notna()]
        
        # 4. 去除含金属或罕见原子的分子
        allowed_atoms = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I
        
        def has_only_common_atoms(mol):
            if mol is None:
                return False
            atoms = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
            return atoms.issubset(allowed_atoms)
        
        df['common_atoms'] = df['mol'].apply(has_only_common_atoms)
        df = df[df['common_atoms']].copy()
        
        # 5. 去除分子量过大或过小的分子
        df['MW'] = df['mol'].apply(Descriptors.MolWt)
        df = df[(df['MW'] >= 50) & (df['MW'] <= 1000)]
        
        final_count = len(df)
        print(f"   清洗后: {final_count} 条")
        print(f"   移除率: {(initial_count - final_count) / initial_count * 100:.1f}%")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'auto') -> pd.DataFrame:
        """
        处理缺失值
        
        strategy: 'auto', 'drop', 'mean', 'median'
        """
        print(f"\n🔧 处理缺失值...")
        
        # 统计缺失值
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) == 0:
            print("   无缺失值")
            return df
        
        print(f"   发现 {len(missing_cols)} 列有缺失值:")
        for col, count in missing_cols.items():
            print(f"     {col}: {count} ({count/len(df)*100:.1f}%)")
        
        if strategy == 'drop':
            # 删除含缺失值的行
            df = df.dropna()
            print(f"   删除后: {len(df)} 条")
            
        elif strategy in ['mean', 'median']:
            # 数值列填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df[col].mean()
                    else:
                        fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"   {col}: 填充为 {strategy} = {fill_value:.3f}")
        
        else:  # auto
            # 自动策略：缺失少的删除行，缺失多的填充
            for col in missing_cols.index:
                if missing_cols[col] / len(df) < 0.05:  # <5% 删除行
                    df = df[df[col].notna()]
                elif df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame, 
                       target_col: str = 'class',
                       method: str = 'undersample') -> pd.DataFrame:
        """
        数据平衡处理
        
        method: 'undersample' (欠采样), 'oversample' (过采样), 'smote' (SMOTE)
        """
        print(f"\n⚖️  数据平衡 ({method})...")
        
        # 统计类别分布
        class_counts = df[target_col].value_counts()
        print(f"   原始分布:")
        for cls, count in class_counts.items():
            print(f"     {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        if len(class_counts) < 2:
            print("   只有一类，无需平衡")
            return df
        
        if method == 'undersample':
            # 欠采样：减少多数类
            min_count = class_counts.min()
            balanced_dfs = []
            for cls in class_counts.index:
                cls_df = df[df[target_col] == cls].sample(min_count, random_state=42)
                balanced_dfs.append(cls_df)
            df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)  # 打乱
            
        elif method == 'oversample':
            # 过采样：复制少数类
            max_count = class_counts.max()
            balanced_dfs = []
            for cls in class_counts.index:
                cls_df = df[df[target_col] == cls]
                if len(cls_df) < max_count:
                    cls_df = cls_df.sample(max_count, replace=True, random_state=42)
                balanced_dfs.append(cls_df)
            df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
        
        # 重新统计
        new_counts = df[target_col].value_counts()
        print(f"   平衡后分布:")
        for cls, count in new_counts.items():
            print(f"     {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    # ==================== 3. 数据验证与质量评估 ====================
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        数据集质量验证
        """
        print(f"\n✅ 数据质量验证...")
        
        report = {
            'total_molecules': len(df),
            'valid_smiles': 0,
            'unique_smiles': 0,
            'valid_molecules': 0,
            'mw_range': {},
            'logp_range': {},
            'lipinski_pass_rate': 0,
            'issues': []
        }
        
        # 1. SMILES 验证
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
        report['valid_smiles'] = df['mol'].notna().sum()
        
        if report['valid_smiles'] < len(df):
            report['issues'].append(f"{len(df) - report['valid_smiles']} 个无效SMILES")
        
        # 2. 唯一性
        report['unique_smiles'] = df['smiles'].nunique()
        if report['unique_smiles'] < len(df):
            report['issues'].append(f"{len(df) - report['unique_smiles']} 个重复SMILES")
        
        # 3. 有效分子
        valid_df = df[df['mol'].notna()].copy()
        report['valid_molecules'] = len(valid_df)
        
        if len(valid_df) == 0:
            report['issues'].append("没有有效的分子")
            return report
        
        # 4. 分子量分布
        valid_df['MW'] = valid_df['mol'].apply(Descriptors.MolWt)
        report['mw_range'] = {
            'min': float(valid_df['MW'].min()),
            'max': float(valid_df['MW'].max()),
            'mean': float(valid_df['MW'].mean())
        }
        
        # 5. LogP 分布
        valid_df['LogP'] = valid_df['mol'].apply(Crippen.MolLogP)
        report['logp_range'] = {
            'min': float(valid_df['LogP'].min()),
            'max': float(valid_df['LogP'].max()),
            'mean': float(valid_df['LogP'].mean())
        }
        
        # 6. Lipinski 通过率
        valid_df['HBD'] = valid_df['mol'].apply(Lipinski.NumHDonors)
        valid_df['HBA'] = valid_df['mol'].apply(Lipinski.NumHAcceptors)
        
        def check_lipinski(row):
            violations = 0
            if row['MW'] > 500: violations += 1
            if row['LogP'] > 5: violations += 1
            if row['HBD'] > 5: violations += 1
            if row['HBA'] > 10: violations += 1
            return violations <= 1
        
        valid_df['lipinski_pass'] = valid_df.apply(check_lipinski, axis=1)
        report['lipinski_pass_rate'] = float(valid_df['lipinski_pass'].mean())
        
        # 7. 活性分布（如果有）
        if 'activity' in df.columns:
            report['activity_stats'] = {
                'min': float(df['activity'].min()),
                'max': float(df['activity'].max()),
                'mean': float(df['activity'].mean()),
                'std': float(df['activity'].std())
            }
        
        # 打印报告
        print(f"\n📊 数据质量报告:")
        print(f"   总分子数: {report['total_molecules']}")
        print(f"   有效SMILES: {report['valid_smiles']}/{report['total_molecules']}")
        print(f"   唯一SMILES: {report['unique_smiles']}")
        print(f"   分子量范围: {report['mw_range']['min']:.1f} - {report['mw_range']['max']:.1f}")
        print(f"   LogP范围: {report['logp_range']['min']:.2f} - {report['logp_range']['max']:.2f}")
        print(f"   Lipinski通过率: {report['lipinski_pass_rate']*100:.1f}%")
        
        if report['issues']:
            print(f"\n⚠️  发现问题:")
            for issue in report['issues']:
                print(f"   - {issue}")
        else:
            print(f"\n✅ 数据质量良好")
        
        # 保存报告 (处理numpy类型)
        report_file = f"{self.output_dir}/data/validation_report.json"
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        report_serializable = json.loads(json.dumps(report, default=convert_to_native))
        with open(report_file, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        print(f"\n   报告保存: {report_file}")
        
        return report
    
    def generate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成数据统计摘要
        """
        print(f"\n📈 生成统计摘要...")
        
        stats = {}
        
        # 基础统计
        stats['分子数量'] = len(df)
        stats['唯一分子'] = df['smiles'].nunique() if 'smiles' in df.columns else 'N/A'
        
        # 分子属性统计
        if 'mol' in df.columns:
            mols = df['mol'].dropna()
            
            stats['平均分子量'] = np.mean([Descriptors.MolWt(m) for m in mols])
            stats['平均LogP'] = np.mean([Crippen.MolLogP(m) for m in mols])
            stats['平均TPSA'] = np.mean([Descriptors.TPSA(m) for m in mols])
        
        # 类别分布
        if 'class' in df.columns:
            class_dist = df['class'].value_counts().to_dict()
            stats['类别分布'] = class_dist
        
        # 活性统计
        if 'activity' in df.columns:
            stats['活性均值'] = df['activity'].mean()
            stats['活性标准差'] = df['activity'].std()
        
        # 打印
        print(f"\n📊 数据摘要:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return pd.DataFrame([stats])


# ==================== 使用示例 ====================

def demo_data_enhancement():
    """
    演示数据增强完整流程
    """
    print("=" * 70)
    print("🧪 药学研究 AI 工作流 - 阶段一：数据增强与验证")
    print("=" * 70)
    
    # 初始化
    enhancer = DataEnhancement(output_dir="./pharma_demo")
    
    # 步骤 1: 创建示例数据集
    print("\n" + "=" * 70)
    print("步骤 1: 创建示例数据集")
    print("=" * 70)
    
    df = enhancer.create_sample_dataset(dataset_type='balanced')
    
    # 步骤 2: 数据清洗
    print("\n" + "=" * 70)
    print("步骤 2: 数据清洗")
    print("=" * 70)
    
    df_clean = enhancer.clean_dataset(df)
    
    # 步骤 3: 处理缺失值
    print("\n" + "=" * 70)
    print("步骤 3: 处理缺失值")
    print("=" * 70)
    
    df_clean = enhancer.handle_missing_values(df_clean, strategy='auto')
    
    # 步骤 4: 数据平衡
    print("\n" + "=" * 70)
    print("步骤 4: 数据平衡")
    print("=" * 70)
    
    df_balanced = enhancer.balance_dataset(df_clean, target_col='class', method='undersample')
    
    # 步骤 5: 数据验证
    print("\n" + "=" * 70)
    print("步骤 5: 数据质量验证")
    print("=" * 70)
    
    validation_report = enhancer.validate_dataset(df_balanced)
    
    # 步骤 6: 统计摘要
    print("\n" + "=" * 70)
    print("步骤 6: 生成统计摘要")
    print("=" * 70)
    
    stats_df = enhancer.generate_statistics(df_balanced)
    
    # 保存最终数据集
    final_file = f"{enhancer.output_dir}/data/final_clean_dataset.csv"
    df_balanced.to_csv(final_file, index=False)
    
    print("\n" + "=" * 70)
    print("✅ 阶段一完成!")
    print(f"📁 最终数据集: {final_file}")
    print(f"📊 数据规模: {len(df_balanced)} 个分子")
    print("=" * 70)
    
    return df_balanced, validation_report


if __name__ == "__main__":
    df, report = demo_data_enhancement()
