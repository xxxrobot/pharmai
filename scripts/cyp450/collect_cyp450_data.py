#!/usr/bin/env python3
"""
CYP450抑制数据收集脚本

从公开资源收集CYP450抑制数据：
- CYP3A4, CYP2D6, CYP2C9 抑制数据
- 使用ChEMBL API获取数据
- 保存为CSV格式
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

# 数据目录
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cyp450', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# CYP450靶点ID (来自ChEMBL)
CYP_TARGETS = {
    'CYP3A4': 'CHEMBL340',  # Cytochrome P450 3A4
    'CYP2D6': 'CHEMBL289',  # Cytochrome P450 2D6
    'CYP2C9': 'CHEMBL335',  # Cytochrome P450 2C9
}

def fetch_chembl_activity(target_id, limit=200):
    """
    从ChEMBL获取活性数据
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    
    params = {
        'target_chembl_id': target_id,
        'standard_type__iregex': '^(IC50|Ki|Kd)$',  # 抑制常数
        'limit': limit,
        'offset': 0
    }
    
    print(f"正在获取靶点 {target_id} 的数据...")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        activities = []
        for item in data.get('activities', []):
            activity = {
                'molecule_chembl_id': item.get('molecule_chembl_id', ''),
                'canonical_smiles': item.get('canonical_smiles', ''),
                'standard_value': item.get('standard_value', ''),
                'standard_units': item.get('standard_units', ''),
                'standard_type': item.get('standard_type', ''),
                'pchembl_value': item.get('pchembl_value', ''),
                'assay_chembl_id': item.get('assay_chembl_id', ''),
                'document_chembl_id': item.get('document_chembl_id', ''),
            }
            activities.append(activity)
        
        print(f"  获取到 {len(activities)} 条记录")
        return activities
        
    except Exception as e:
        print(f"  获取数据失败: {e}")
        return []

def create_cyp450_dataset():
    """
    创建CYP450数据集
    """
    all_data = []
    
    for cyp_name, target_id in CYP_TARGETS.items():
        print(f"\n处理 {cyp_name} ({target_id})...")
        
        # 获取活性数据
        activities = fetch_chembl_activity(target_id, limit=100)
        
        for activity in activities:
            record = {
                'cyp_isoform': cyp_name,
                'target_id': target_id,
                'molecule_id': activity['molecule_chembl_id'],
                'smiles': activity['canonical_smiles'],
                'activity_value': activity['standard_value'],
                'activity_units': activity['standard_units'],
                'activity_type': activity['standard_type'],
                'pchembl_value': activity['pchembl_value'],
                'assay_id': activity['assay_chembl_id'],
                'document_id': activity['document_chembl_id'],
            }
            all_data.append(record)
        
        # 避免请求过快
        time.sleep(1)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 保存数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(RAW_DATA_DIR, f'cyp450_raw_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n数据收集完成!")
    print(f"总计收集: {len(df)} 条记录")
    print(f"文件保存到: {output_file}")
    
    # 统计数据
    print("\n按CYP亚型统计:")
    print(df['cyp_isoform'].value_counts())
    
    return df

def process_cyp450_data(df):
    """
    处理CYP450数据，创建训练集
    """
    if df.empty:
        print("没有数据需要处理")
        return None
    
    # 数据清洗
    print("\n开始数据处理...")
    
    # 1. 移除缺失SMILES的记录
    df = df.dropna(subset=['smiles'])
    print(f"  移除缺失SMILES后: {len(df)} 条")
    
    # 2. 转换为标准单位 (nM)
    def convert_to_nm(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return None
        
        value = float(value)
        unit = str(unit).lower()
        
        if unit == 'nm':
            return value
        elif unit == 'μm' or unit == 'um':
            return value * 1000
        elif unit == 'mm':
            return value * 1000000
        else:
            return None
    
    df['activity_nM'] = df.apply(
        lambda row: convert_to_nm(row['activity_value'], row['activity_units']),
        axis=1
    )
    
    # 3. 创建二分类标签 (抑制/非抑制)
    # 假设IC50/Ki < 10μM 为抑制
    df['is_inhibitor'] = df['activity_nM'].apply(
        lambda x: 1 if x is not None and x < 10000 else 0
    )
    
    # 4. 移除转换失败或无效的记录
    df = df.dropna(subset=['activity_nM'])
    print(f"  有效数值记录: {len(df)} 条")
    
    # 5. 按CYP亚型分开
    processed_data = {}
    for cyp in df['cyp_isoform'].unique():
        cyp_df = df[df['cyp_isoform'] == cyp].copy()
        print(f"  {cyp}: {len(cyp_df)} 条记录, 抑制剂: {cyp_df['is_inhibitor'].sum()}")
        processed_data[cyp] = cyp_df
    
    # 保存处理后的数据
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cyp450', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for cyp, cyp_df in processed_data.items():
        output_file = os.path.join(PROCESSED_DIR, f'{cyp.lower()}_processed_{timestamp}.csv')
        cyp_df.to_csv(output_file, index=False)
        print(f"  保存 {cyp} 数据到: {output_file}")
    
    return processed_data

if __name__ == "__main__":
    print("=" * 60)
    print("CYP450抑制数据收集脚本")
    print("=" * 60)
    
    # 收集数据
    raw_df = create_cyp450_dataset()
    
    if raw_df is not None and not raw_df.empty:
        # 处理数据
        processed_data = process_cyp450_data(raw_df)
        
        # 生成摘要报告
        print("\n" + "=" * 60)
        print("数据收集摘要:")
        print("=" * 60)
        
        total_records = len(raw_df)
        processed_records = sum(len(df) for df in processed_data.values()) if processed_data else 0
        
        print(f"原始记录总数: {total_records}")
        print(f"处理后记录数: {processed_records}")
        print(f"数据保留率: {(processed_records/total_records*100):.1f}%")
        
        if processed_data:
            for cyp, cyp_df in processed_data.items():
                inhibitors = cyp_df['is_inhibitor'].sum()
                total = len(cyp_df)
                print(f"{cyp}: {total} 条记录, 抑制剂: {inhibitors} ({inhibitors/total*100:.1f}%)")
    else:
        print("\n没有收集到数据，请检查网络连接或ChEMBL API状态")
    
    print("\n脚本执行完成!")
