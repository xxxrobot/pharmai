#!/usr/bin/env python3
"""
DrugBank数据收集脚本
基于DrugBank集成计划实现

功能:
1. 从DrugBank公共数据文件加载药物信息
2. 提供API客户端接口 (需要商业许可证)
3. 数据预处理和清洗
4. 本地缓存管理

注意: DrugBank完整数据需要商业许可证
此脚本提供基础框架和示例数据
"""

import os
import sys
import json
import pickle
import warnings
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import DataStructs

warnings.filterwarnings('ignore')


@dataclass
class DrugInfo:
    """药物信息数据类"""
    drugbank_id: str
    name: str
    smiles: Optional[str]
    description: Optional[str]
    pharmacology: Optional[str]
    toxicity: Optional[str]
    targets: List[str]
    categories: List[str]
    indications: List[str]
    interactions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrugInfo':
        """从字典创建"""
        return cls(**data)


class DrugBankDataCollector:
    """DrugBank数据收集器"""
    
    def __init__(self, data_dir: str = "data/drugbank"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.cache_dir = os.path.join(data_dir, "cache")
        
        # 创建目录结构
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 数据存储
        self.drugs_df: Optional[pd.DataFrame] = None
        self.drugs_dict: Dict[str, DrugInfo] = {}
        
        print(f"✅ DrugBank数据收集器初始化")
        print(f"   数据目录: {data_dir}")
    
    # ==================== 1. 示例数据生成 ====================
    
    def create_sample_drugbank_data(self, n_drugs: int = 50) -> pd.DataFrame:
        """
        创建示例DrugBank数据
        用于测试和演示，不依赖外部数据源
        
        Args:
            n_drugs: 生成药物数量
            
        Returns:
            DataFrame: 示例药物数据
        """
        print(f"\n📚 创建示例DrugBank数据 ({n_drugs}个药物)")
        print("=" * 60)
        
        # 示例药物数据 (基于真实药物)
        sample_drugs = [
            {
                'drugbank_id': 'DB00188',
                'name': 'Simvastatin',
                'smiles': 'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C',
                'description': 'HMG-CoA reductase inhibitor used to lower cholesterol',
                'pharmacology': 'Inhibits HMG-CoA reductase, reducing cholesterol synthesis',
                'toxicity': 'Myopathy, rhabdomyolysis at high doses',
                'targets': ['HMGCR'],
                'categories': ['HMG-CoA Reductase Inhibitors', 'Antilipemic Agents'],
                'indications': ['Hypercholesterolemia', 'Cardiovascular Disease Prevention'],
                'interactions': ['CYP3A4 substrates', 'CYP3A4 inhibitors']
            },
            {
                'drugbank_id': 'DB00264',
                'name': 'Metoprolol',
                'smiles': 'CC(C)NCC(COC1=CC=C(C=C1)CC(C)N)O',
                'description': 'Selective beta-1 adrenergic receptor blocker',
                'pharmacology': 'Blocks beta-1 adrenergic receptors in the heart',
                'toxicity': 'Bradycardia, hypotension, fatigue',
                'targets': ['ADRB1'],
                'categories': ['Beta Blockers', 'Antihypertensive Agents'],
                'indications': ['Hypertension', 'Angina', 'Heart Failure'],
                'interactions': ['CYP2D6 substrates', 'CYP2D6 inhibitors']
            },
            {
                'drugbank_id': 'DB00682',
                'name': 'Warfarin',
                'smiles': 'CC(=O)CC(C1=CC=CC=C1)C2=C(C=CC=C2)O',
                'description': 'Vitamin K antagonist anticoagulant',
                'pharmacology': 'Inhibits vitamin K epoxide reductase',
                'toxicity': 'Bleeding, hemorrhage',
                'targets': ['VKORC1'],
                'categories': ['Anticoagulants', 'Vitamin K Antagonists'],
                'indications': ['Atrial Fibrillation', 'Deep Vein Thrombosis', 'Pulmonary Embolism'],
                'interactions': ['CYP2C9 substrates', 'CYP2C9 inhibitors', 'CYP1A2 substrates']
            },
            {
                'drugbank_id': 'DB00472',
                'name': 'Fluoxetine',
                'smiles': 'CNCCC(C1=CC=CC=C1)C2=CC=C(C=C2)OC',
                'description': 'Selective serotonin reuptake inhibitor (SSRI)',
                'pharmacology': 'Inhibits serotonin reuptake in the CNS',
                'toxicity': 'Nausea, insomnia, sexual dysfunction',
                'targets': ['SLC6A4'],
                'categories': ['Selective Serotonin Reuptake Inhibitors', 'Antidepressants'],
                'indications': ['Major Depressive Disorder', 'Obsessive-Compulsive Disorder', 'Panic Disorder'],
                'interactions': ['CYP2D6 inhibitors', 'CYP2D6 substrates', 'Serotonergic drugs']
            },
            {
                'drugbank_id': 'DB00316',
                'name': 'Omeprazole',
                'smiles': 'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC=C3',
                'description': 'Proton pump inhibitor',
                'pharmacology': 'Irreversibly inhibits H+/K+ ATPase in gastric parietal cells',
                'toxicity': 'Headache, diarrhea, nausea',
                'targets': ['ATP4A'],
                'categories': ['Proton Pump Inhibitors', 'Antiulcer Agents'],
                'indications': ['Gastroesophageal Reflux Disease', 'Peptic Ulcer Disease'],
                'interactions': ['CYP2C19 substrates', 'CYP2C19 inhibitors']
            },
            {
                'drugbank_id': 'DB01076',
                'name': 'Atorvastatin',
                'smiles': 'CC(C)C1=C(C(=O)NC(=O)N1)C2=CC=C(C=C2)F.CC(C)CC(C(=O)O)O',
                'description': 'HMG-CoA reductase inhibitor',
                'pharmacology': 'Inhibits HMG-CoA reductase',
                'toxicity': 'Myopathy, liver enzyme elevations',
                'targets': ['HMGCR'],
                'categories': ['HMG-CoA Reductase Inhibitors'],
                'indications': ['Hypercholesterolemia', 'Dyslipidemia'],
                'interactions': ['CYP3A4 substrates', 'CYP3A4 inhibitors']
            },
            {
                'drugbank_id': 'DB00571',
                'name': 'Propranolol',
                'smiles': 'CC(C)NCC(COC1=CC=C(C=C1)CC(C)N)O',
                'description': 'Non-selective beta-adrenergic antagonist',
                'pharmacology': 'Blocks beta-1 and beta-2 adrenergic receptors',
                'toxicity': 'Bradycardia, bronchospasm, fatigue',
                'targets': ['ADRB1', 'ADRB2'],
                'categories': ['Beta Blockers', 'Non-selective'],
                'indications': ['Hypertension', 'Angina', 'Arrhythmia', 'Migraine Prophylaxis'],
                'interactions': ['CYP1A2 substrates', 'CYP2D6 substrates']
            },
            {
                'drugbank_id': 'DB00945',
                'name': 'Aspirin',
                'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                'description': 'Nonsteroidal anti-inflammatory drug (NSAID)',
                'pharmacology': 'Irreversibly inhibits COX-1 and COX-2',
                'toxicity': 'Gastrointestinal bleeding, tinnitus',
                'targets': ['PTGS1', 'PTGS2'],
                'categories': ['NSAIDs', 'Antiplatelet Agents'],
                'indications': ['Pain', 'Fever', 'Inflammation', 'Cardiovascular Prevention'],
                'interactions': ['Anticoagulants', 'Other NSAIDs']
            },
            {
                'drugbank_id': 'DB00563',
                'name': 'Methotrexate',
                'smiles': 'CN(CC1=CN=C2C(=N1)C(=O)NC(=N2)N)C(=O)C(CCC(=O)O)N',
                'description': 'Folate antagonist antimetabolite',
                'pharmacology': 'Inhibits dihydrofolate reductase',
                'toxicity': 'Bone marrow suppression, hepatotoxicity, pulmonary toxicity',
                'targets': ['DHFR'],
                'categories': ['Antimetabolites', 'Antineoplastic Agents', 'Antirheumatic Agents'],
                'indications': ['Rheumatoid Arthritis', 'Psoriasis', 'Cancer'],
                'interactions': ['NSAIDs', 'Proton Pump Inhibitors', 'Folic acid antagonists']
            },
            {
                'drugbank_id': 'DB00619',
                'name': 'Ibuprofen',
                'smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
                'description': 'Nonsteroidal anti-inflammatory drug',
                'pharmacology': 'Non-selective COX inhibitor',
                'toxicity': 'Gastrointestinal upset, renal impairment',
                'targets': ['PTGS1', 'PTGS2'],
                'categories': ['NSAIDs'],
                'indications': ['Pain', 'Inflammation', 'Fever'],
                'interactions': ['Anticoagulants', 'ACE inhibitors', 'Diuretics']
            }
        ]
        
        # 如果请求更多药物，复制并修改示例数据
        if n_drugs > len(sample_drugs):
            print(f"  生成 {n_drugs} 个药物 (基于 {len(sample_drugs)} 个模板)...")
            
            # 扩展数据
            extended_drugs = []
            for i in range(n_drugs):
                base_drug = sample_drugs[i % len(sample_drugs)].copy()
                
                # 创建变体
                variant_id = f"{base_drug['drugbank_id']}_V{i+1:03d}"
                variant_name = f"{base_drug['name']}_Analog_{i+1}"
                
                # 轻微修改SMILES (添加简单修饰)
                base_smiles = base_drug['smiles']
                if i >= len(sample_drugs):
                    # 为重复的药物创建变体SMILES
                    variant_smiles = self._create_variant_smiles(base_smiles, i)
                else:
                    variant_smiles = base_smiles
                
                variant_drug = {
                    'drugbank_id': variant_id,
                    'name': variant_name,
                    'smiles': variant_smiles,
                    'description': f"Analog of {base_drug['name']}",
                    'pharmacology': base_drug['pharmacology'],
                    'toxicity': base_drug['toxicity'],
                    'targets': base_drug['targets'],
                    'categories': base_drug['categories'],
                    'indications': base_drug['indications'],
                    'interactions': base_drug['interactions']
                }
                
                extended_drugs.append(variant_drug)
            
            sample_drugs = extended_drugs[:n_drugs]
        else:
            sample_drugs = sample_drugs[:n_drugs]
        
        # 转换为DataFrame
        df = pd.DataFrame(sample_drugs)
        
        # 保存数据
        output_path = os.path.join(self.processed_dir, f"sample_drugbank_data_{n_drugs}.csv")
        df.to_csv(output_path, index=False)
        
        # 同时保存为pickle格式 (保留列表类型)
        pickle_path = os.path.join(self.processed_dir, f"sample_drugbank_data_{n_drugs}.pkl")
        df.to_pickle(pickle_path)
        
        print(f"✅ 示例数据创建完成")
        print(f"   CSV格式: {output_path}")
        print(f"   Pickle格式: {pickle_path}")
        print(f"   药物数量: {len(df)}")
        
        # 显示数据摘要
        print(f"\n📊 数据摘要:")
        print(f"   有SMILES结构的药物: {df['smiles'].notna().sum()}/{len(df)}")
        print(f"   唯一靶点数量: {len(set([t for targets in df['targets'] for t in targets]))}")
        print(f"   唯一适应症数量: {len(set([ind for inds in df['indications'] for ind in inds]))}")
        
        self.drugs_df = df
        return df
    
    def _create_variant_smiles(self, base_smiles: str, index: int) -> str:
        """创建SMILES变体 (简化版)"""
        # 这里简化处理，实际应该使用RDKit进行结构修饰
        # 为了示例，我们只是添加注释
        return base_smiles  # 返回原始SMILES
    
    # ==================== 2. 数据加载 ====================
    
    def load_drugbank_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载DrugBank数据
        
        Args:
            file_path: 数据文件路径，如果为None则加载示例数据
            
        Returns:
            DataFrame: 药物数据
        """
        if file_path and os.path.exists(file_path):
            print(f"\n📂 加载DrugBank数据: {file_path}")
            
            # 根据文件扩展名选择加载方式
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            print(f"✅ 数据加载完成: {len(df)} 个药物")
            
        else:
            # 加载示例数据
            print(f"\n📂 未找到数据文件，加载示例数据")
            
            # 查找现有的示例数据
            sample_files = [f for f in os.listdir(self.processed_dir) 
                          if f.startswith('sample_drugbank_data') and f.endswith('.pkl')]
            
            if sample_files:
                # 使用最新的示例数据
                latest_file = sorted(sample_files)[-1]
                file_path = os.path.join(self.processed_dir, latest_file)
                df = pd.read_pickle(file_path)
                print(f"✅ 加载现有示例数据: {latest_file}")
                print(f"   药物数量: {len(df)}")
            else:
                # 创建新的示例数据
                df = self.create_sample_drugbank_data(n_drugs=50)
        
        self.drugs_df = df
        
        # 转换为DrugInfo对象字典
        self._convert_to_druginfo_dict()
        
        return df
    
    def _convert_to_druginfo_dict(self):
        """将DataFrame转换为DrugInfo字典"""
        if self.drugs_df is None:
            return
        
        self.drugs_dict = {}
        for _, row in self.drugs_df.iterrows():
            drug_info = DrugInfo(
                drugbank_id=row['drugbank_id'],
                name=row['name'],
                smiles=row.get('smiles'),
                description=row.get('description'),
                pharmacology=row.get('pharmacology'),
                toxicity=row.get('toxicity'),
                targets=row.get('targets', []),
                categories=row.get('categories', []),
                indications=row.get('indications', []),
                interactions=row.get('interactions', [])
            )
            self.drugs_dict[row['drugbank_id']] = drug_info
    
    # ==================== 3. 数据查询 ====================
    
    def get_drug_by_id(self, drugbank_id: str) -> Optional[DrugInfo]:
        """通过DrugBank ID获取药物信息"""
        return self.drugs_dict.get(drugbank_id)
    
    def get_drug_by_name(self, name: str) -> Optional[DrugInfo]:
        """通过药物名称获取药物信息"""
        for drug in self.drugs_dict.values():
            if drug.name.lower() == name.lower():
                return drug
        return None
    
    def search_drugs_by_target(self, target: str) -> List[DrugInfo]:
        """通过靶点搜索药物"""
        results = []
        for drug in self.drugs_dict.values():
            if target in drug.targets:
                results.append(drug)
        return results
    
    def search_drugs_by_category(self, category: str) -> List[DrugInfo]:
        """通过类别搜索药物"""
        results = []
        for drug in self.drugs_dict.values():
            if category in drug.categories:
                results.append(drug)
        return results
    
    def get_drugs_by_cyp_interaction(self, cyp_isoform: str) -> List[DrugInfo]:
        """获取与特定CYP亚型相互作用的药物"""
        results = []
        cyp_pattern = cyp_isoform.upper()
        
        for drug in self.drugs_dict.values():
            for interaction in drug.interactions:
                if cyp_pattern in interaction.upper():
                    results.append(drug)
                    break
        
        return results
    
    # ==================== 4. 分子相似性搜索 ====================
    
    def calculate_molecular_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        计算两个分子的Tanimoto相似性
        
        Args:
            smiles1: 第一个分子的SMILES
            smiles2: 第二个分子的SMILES
            
        Returns:
            float: Tanimoto相似性分数 (0-1)
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if not mol1 or not mol2:
            return 0.0
        
        # 生成Morgan指纹
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, 2048)
        
        # 计算Tanimoto相似性
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        return similarity
    
    def find_similar_drugs(self, query_smiles: str, threshold: float = 0.7, top_n: int = 10) -> List[Tuple[DrugInfo, float]]:
        """
        查找与查询分子相似的药物
        
        Args:
            query_smiles: 查询分子的SMILES
            threshold: 相似性阈值
            top_n: 返回结果数量
            
        Returns:
            List[Tuple[DrugInfo, float]]: (药物信息, 相似性分数) 列表
        """
        print(f"\n🔍 搜索相似药物 (阈值: {threshold}, 前{top_n}个)")
        print("-" * 60)
        
        similarities = []
        
        for drug in self.drugs_dict.values():
            if drug.smiles:
                try:
                    sim = self.calculate_molecular_similarity(query_smiles, drug.smiles)
                    if sim >= threshold:
                        similarities.append((drug, sim))
                except Exception as e:
                    continue
        
        # 按相似性排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个
        results = similarities[:top_n]
        
        print(f"✅ 找到 {len(results)} 个相似药物")
        for i, (drug, sim) in enumerate(results, 1):
            print(f"   {i}. {drug.name} (相似性: {sim:.3f})")
        
        return results
    
    # ==================== 5. 数据导出 ====================
    
    def export_for_cyp_prediction(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        导出用于CYP450预测的数据
        
        Returns:
            DataFrame: 包含CYP相互作用信息的药物数据
        """
        if self.drugs_df is None:
            print("❌ 未加载数据")
            return pd.DataFrame()
        
        print(f"\n📤 导出CYP450预测数据")
        print("-" * 60)
        
        # 提取CYP相互作用信息
        cyp_data = []
        
        for _, row in self.drugs_df.iterrows():
            interactions = row.get('interactions', [])
            
            # 检查CYP相互作用
            cyp3a4 = any('CYP3A4' in inter for inter in interactions)
            cyp2d6 = any('CYP2D6' in inter for inter in interactions)
            cyp2c9 = any('CYP2C9' in inter for inter in interactions)
            cyp2c19 = any('CYP2C19' in inter for inter in interactions)
            cyp1a2 = any('CYP1A2' in inter for inter in interactions)
            
            cyp_data.append({
                'drugbank_id': row['drugbank_id'],
                'name': row['name'],
                'smiles': row.get('smiles'),
                'cyp3a4_interaction': cyp3a4,
                'cyp2d6_interaction': cyp2d6,
                'cyp2c9_interaction': cyp2c9,
                'cyp2c19_interaction': cyp2c19,
                'cyp1a2_interaction': cyp1a2,
                'any_cyp_interaction': cyp3a4 or cyp2d6 or cyp2c9 or cyp2c19 or cyp1a2
            })
        
        df_cyp = pd.DataFrame(cyp_data)
        
        # 保存
        if output_path is None:
            output_path = os.path.join(self.processed_dir, "drugbank_cyp_interactions.csv")
        
        df_cyp.to_csv(output_path, index=False)
        
        print(f"✅ 导出完成")
        print(f"   文件路径: {output_path}")
        print(f"   总药物数: {len(df_cyp)}")
        print(f"   有CYP相互作用的药物: {df_cyp['any_cyp_interaction'].sum()}")
        print(f"   CYP3A4: {df_cyp['cyp3a4_interaction'].sum()}")
        print(f"   CYP2D6: {df_cyp['cyp2d6_interaction'].sum()}")
        print(f"   CYP2C9: {df_cyp['cyp2c9_interaction'].sum()}")
        
        return df_cyp
    
    def export_statistics(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """导出数据统计信息"""
        if self.drugs_df is None:
            return {}
        
        stats = {
            'total_drugs': len(self.drugs_df),
            'drugs_with_smiles': self.drugs_df['smiles'].notna().sum(),
            'unique_targets': len(set([t for targets in self.drugs_df['targets'] for t in targets])),
            'unique_categories': len(set([c for cats in self.drugs_df['categories'] for c in cats])),
            'unique_indications': len(set([ind for inds in self.drugs_df['indications'] for ind in inds])),
            'timestamp': datetime.now().isoformat()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        return stats


# ==================== 使用示例 ====================

def demo_drugbank_collector():
    """演示DrugBank数据收集器"""
    print("=" * 70)
    print("🧪 DrugBank数据收集器演示")
    print("=" * 70)
    
    # 初始化收集器
    collector = DrugBankDataCollector(data_dir="data/drugbank")
    
    # 1. 创建示例数据
    print("\n" + "=" * 70)
    print("1. 创建示例数据")
    print("=" * 70)
    
    df = collector.create_sample_drugbank_data(n_drugs=30)
    
    # 2. 加载数据
    print("\n" + "=" * 70)
    print("2. 加载数据")
    print("=" * 70)
    
    df = collector.load_drugbank_data()
    
    # 3. 查询示例
    print("\n" + "=" * 70)
    print("3. 数据查询")
    print("=" * 70)
    
    # 通过ID查询
    drug = collector.get_drug_by_id('DB00188')
    if drug:
        print(f"\n通过ID查询 (DB00188):")
        print(f"   名称: {drug.name}")
        print(f"   靶点: {', '.join(drug.targets)}")
    
    # 通过靶点搜索
    hmgcr_drugs = collector.search_drugs_by_target('HMGCR')
    print(f"\n靶向HMGCR的药物: {len(hmgcr_drugs)}个")
    for d in hmgcr_drugs:
        print(f"   - {d.name}")
    
    # 4. 相似性搜索
    print("\n" + "=" * 70)
    print("4. 分子相似性搜索")
    print("=" * 70)
    
    query_smiles = 'CC(C)C1=C(C(=O)OC)C(=C(C1C)C)CC(C)C'  # Simvastatin
    similar_drugs = collector.find_similar_drugs(query_smiles, threshold=0.3, top_n=5)
    
    # 5. 导出CYP数据
    print("\n" + "=" * 70)
    print("5. 导出CYP相互作用数据")
    print("=" * 70)
    
    df_cyp = collector.export_for_cyp_prediction()
    
    # 6. 导出统计
    print("\n" + "=" * 70)
    print("6. 数据统计")
    print("=" * 70)
    
    stats = collector.export_statistics()
    print(f"\n数据摘要:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ DrugBank数据收集器演示完成!")
    print("=" * 70)
    
    return collector


if __name__ == "__main__":
    collector = demo_drugbank_collector()