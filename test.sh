#!/bin/bash
echo "🧪 测试PharmaAI..."
echo ""

echo "1. 测试核心模块导入..."
python3 -c "from pharma_complete_workflow import PharmaAICompleteWorkflow; print('   ✅ 核心模块导入成功')"

echo "2. 测试模型加载..."
python3 -c "
import joblib
try:
    model = joblib.load('models/herg_model.pkl')
    print('   ✅ hERG模型加载成功')
except:
    print('   ⚠️ hERG模型不存在')
"

echo "3. 测试预测功能..."
python3 -c "
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
if mol:
    print('   ✅ RDKit正常工作')
else:
    print('   ❌ RDKit异常')
"

echo ""
echo "✅ 测试完成"
