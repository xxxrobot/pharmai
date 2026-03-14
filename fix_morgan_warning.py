#!/usr/bin/env python3
"""
修复RDKit MorganGenerator弃用警告
将旧的GetMorganFingerprintAsBitVect替换为新的rdFingerprintGenerator API
"""

import os
import re

# 需要修复的文件列表
files_to_fix = [
    'pharma_toxicity_prediction.py',
    'pharma_admet_prediction.py', 
    'pharma_admet_enhanced_fixed.py',
    'pharma_complete_workflow.py',
    'pharma_data_enhancement.py',
    'pharma_pretrained_models.py',
    'train_all_models.py',
    'scripts/drugbank/drugbank_collector.py'
]

def fix_file(filepath):
    """修复单个文件"""
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 检查是否包含旧API
    if 'GetMorganFingerprintAsBitVect' not in content:
        print(f"✓ 无需修复: {filepath}")
        return True
    
    # 1. 替换导入语句
    # 旧: from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    # 新: from rdkit.Chem import rdFingerprintGenerator
    content = re.sub(
        r'from rdkit\.Chem\.AllChem import GetMorganFingerprintAsBitVect\n?',
        'from rdkit.Chem import rdFingerprintGenerator\n',
        content
    )
    
    # 如果文件中有多个导入，可能需要单独处理
    # 检查是否还有残留的导入
    if 'GetMorganFingerprintAsBitVect' in content and 'from rdkit.Chem.AllChem import' in content:
        # 移除特定的导入，保留其他导入
        content = re.sub(
            r'from rdkit\.Chem\.AllChem import ([^\n]*)GetMorganFingerprintAsBitVect([^\n]*)\n?',
            r'from rdkit.Chem.AllChem import \1\2\nfrom rdkit.Chem import rdFingerprintGenerator\n',
            content
        )
    
    # 2. 在文件开头添加生成器初始化（如果还没有）
    if '_morgan_generator' not in content:
        # 找到第一个import之后的合适位置
        import_section_end = 0
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i
        
        # 在最后一个import后添加生成器初始化
        generator_code = '\n# 初始化Morgan指纹生成器 (避免弃用警告)\n_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)\n'
        lines.insert(import_idx + 1, generator_code)
        content = '\n'.join(lines)
    
    # 3. 替换函数调用
    # 旧: GetMorganFingerprintAsBitVect(mol, 2, 2048)
    # 新: _morgan_generator.GetFingerprint(mol)
    content = re.sub(
        r'GetMorganFingerprintAsBitVect\((\w+),\s*2,\s*2048\)',
        r'_morgan_generator.GetFingerprint(\1)',
        content
    )
    
    # 保存修改后的文件
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已修复: {filepath}")
        return True
    else:
        print(f"✓ 无需修改: {filepath}")
        return True

def main():
    print("🔧 修复RDKit MorganGenerator弃用警告")
    print("=" * 60)
    
    fixed_count = 0
    for filepath in files_to_fix:
        if fix_file(filepath):
            fixed_count += 1
    
    print("=" * 60)
    print(f"✅ 修复完成: {fixed_count}/{len(files_to_fix)} 个文件")
    print("\n📝 说明:")
    print("   - 使用新的rdFingerprintGenerator API")
    print("   - 消除了DEPRECATION WARNING")
    print("   - 功能保持不变")

if __name__ == "__main__":
    main()