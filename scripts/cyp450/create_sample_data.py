#!/usr/bin/env python3
"""
CYP450示例数据生成器
当ChEMBL API不可用时，使用示例数据继续开发
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_sample_cyp450_data():
    """创建CYP450示例数据集"""
    
    # 示例化合物（已知的CYP抑制剂和非抑制剂）
    sample_compounds = [
        # CYP3A4抑制剂 (真实药物)
        {'smiles': 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'name': 'Caffeine', 'cyp3a4_inhibitor': 0},
        {'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'name': 'Ibuprofen', 'cyp3a4_inhibitor': 0},
        {'smiles': 'CN1CCOC(=O)C(c2ccccc2)c2ccccc21', 'name': 'Ketoconazole', 'cyp3a4_inhibitor': 1},
        {'smiles': 'CC(C)(C)NCC(O)c1ccc(O)cc1', 'name': 'Ritonavir', 'cyp3a4_inhibitor': 1},
        {'smiles': 'COC1=CC2=C(C=C1OC)C(=O)C=C(O2)C3=CC=CC=C3', 'name': 'Flavonoid', 'cyp3a4_inhibitor': 1},
        
        # CYP2D6底物/抑制剂
        {'smiles': 'CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1', 'name': 'Diphenhydramine', 'cyp2d6_inhibitor': 1},
        {'smiles': 'COC1=CC=CC=C1OC', 'name': 'Paroxetine', 'cyp2d6_inhibitor': 1},
        {'smiles': 'CN1CCC2=CC=CC=C2C1', 'name': 'Nicotine', 'cyp2d6_inhibitor': 0},
        
        # CYP2C9底物/抑制剂
        {'smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'name': 'Warfarin', 'cyp2c9_inhibitor': 1},
        {'smiles': 'CC1=CC=C(C=C1)S(=O)(=O)N', 'name': 'Sulfamethoxazole', 'cyp2c9_inhibitor': 0},
    ]
    
    # 扩展数据集
    all_records = []
    compound_id = 1000
    
    for compound in sample_compounds:
        # 为每个CYP亚型创建记录
        for cyp_isoform in ['CYP3A4', 'CYP2D6', 'CYP2C9']:
            # 根据化合物类型确定抑制状态
            inhibitor_key = f'{cyp_isoform.lower()}_inhibitor'
            is_inhibitor = compound.get(inhibitor_key, np.random.choice([0, 1], p=[0.7, 0.3]))
            
            # 生成模拟的IC50值 (nM)
            if is_inhibitor:
                ic50 = np.random.uniform(100, 5000)  # 抑制剂: 100-5000 nM
            else:
                ic50 = np.random.uniform(10000, 100000)  # 非抑制剂: 10-100 μM
            
            record = {
                'compound_id': f'CMPD{compound_id}',
                'cyp_isoform': cyp_isoform,
                'smiles': compound['smiles'],
                'compound_name': compound['name'],
                'ic50_nM': round(ic50, 2),
                'pIC50': round(-np.log10(ic50 * 1e-9), 2),
                'is_inhibitor': int(is_inhibitor),
                'source': 'sample_data',
                'confidence': 'high' if compound.get(inhibitor_key) is not None else 'medium'
            }
            
            all_records.append(record)
            compound_id += 1
    
    # 创建DataFrame
    df = pd.DataFrame(all_records)
    
    return df

def save_sample_data():
    """保存示例数据"""
    
    # 创建目录
    raw_dir = os.path.join('data', 'cyp450', 'raw')
    processed_dir = os.path.join('data', 'cyp450', 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 生成数据
    df = create_sample_cyp450_data()
    
    # 保存原始数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_file = os.path.join(raw_dir, f'cyp450_sample_{timestamp}.csv')
    df.to_csv(raw_file, index=False)
    
    print(f"示例数据已保存到: {raw_file}")
    print(f"记录总数: {len(df)}")
    
    # 按CYP亚型统计
    print("\n按CYP亚型统计:")
    for cyp in df['cyp_isoform'].unique():
        cyp_df = df[df['cyp_isoform'] == cyp]
        inhibitors = cyp_df['is_inhibitor'].sum()
        total = len(cyp_df)
        print(f"  {cyp}: {total} 条记录, 抑制剂: {inhibitors} ({inhibitors/total*100:.1f}%)")
    
    # 保存处理后的数据（按亚型分开）
    for cyp in df['cyp_isoform'].unique():
        cyp_df = df[df['cyp_isoform'] == cyp]
        processed_file = os.path.join(processed_dir, f'{cyp.lower()}_sample_{timestamp}.csv')
        cyp_df.to_csv(processed_file, index=False)
        print(f"  {cyp} 处理数据保存到: {processed_file}")
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("CYP450示例数据生成器")
    print("=" * 60)
    
    df = save_sample_data()
    
    print("\n" + "=" * 60)
    print("数据摘要:")
    print("=" * 60)
    print(df.head(10))
    
    print("\n生成完成！")
