# PharmaAI - Node.js vs Python 技术选型分析

## 🤔 问题核心

**PharmaAI是否应该从Python迁移到Node.js以更好地适配OpenClaw？**

---

## 📊 技术对比

### OpenClaw 技术栈

```
OpenClaw 架构
├── Gateway (Node.js/TypeScript)  ← 核心
├── Agent Runtime (Node.js)       ← 执行环境
├── Skills (多语言支持)           ← 插件系统
│   ├── Node.js skills            ← 原生支持
│   ├── Python skills             ← 通过子进程
│   └── 其他语言                  ← 通过CLI
└── ACP (Agent Communication Protocol)
```

### 对比维度

| 维度 | Python | Node.js | 评估 |
|------|--------|---------|------|
| **OpenClaw原生支持** | ⭐⭐ 通过子进程 | ⭐⭐⭐⭐⭐ 原生 | Node.js胜 |
| **化学信息学生态** | ⭐⭐⭐⭐⭐ RDKit | ⭐⭐ 有限 | Python胜 |
| **机器学习生态** | ⭐⭐⭐⭐⭐ scikit-learn | ⭐⭐⭐ TensorFlow.js | Python胜 |
| **性能** | ⭐⭐⭐ 科学计算快 | ⭐⭐⭐⭐ 异步IO快 | 平手 |
| **启动速度** | ⭐⭐ 较慢 | ⭐⭐⭐⭐⭐ 快 | Node.js胜 |
| **内存占用** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较低 | Node.js胜 |
| **Skill开发体验** | ⭐⭐ 需要包装 | ⭐⭐⭐⭐⭐ 原生 | Node.js胜 |

---

## 🎯 深度分析

### 1. OpenClaw 原生集成

**Node.js 优势：**
```javascript
// Node.js Skill - 直接集成
const { skill } = require('@openclaw/skill-sdk');

skill.define('pharma-predict', async (input) => {
  // 直接调用，无需子进程
  return await predictMolecule(input.smiles);
});
```

**Python 现状：**
```python
# Python Skill - 需要子进程包装
# OpenClaw 需要: node wrapper → spawn python → return result
# 增加延迟 ~100-500ms
```

### 2. 科学计算生态

**Python 优势（难以替代）：**
```python
# RDKit - 化学信息学金标准
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles('CCO')
mw = Descriptors.MolWt(mol)  # 精确分子量
```

**Node.js 替代方案：**
```javascript
// 选项1: 使用rdkit-js (WebAssembly版本)
// 限制: 功能不完整，性能较差

// 选项2: 调用Python子进程
// 缺点: 增加复杂度，失去Node.js优势

// 选项3: 使用纯JS化学库 (如chem.js)
// 缺点: 功能有限，不专业
```

### 3. 机器学习模型

**当前Python模型：**
```python
# scikit-learn 模型
import joblib
model = joblib.load('herg_model.pkl')  # 随机森林
```

**Node.js 迁移选项：**

#### 选项A: ONNX格式转换 ✅ 可行
```python
# Python: 转换为ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 2058]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("herg_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

```javascript
// Node.js: 使用ONNX Runtime
const ort = require('onnxruntime-node');
const session = await ort.InferenceSession.create('herg_model.onnx');
const result = await session.run({ input: tensor });
```

**优点：**
- 模型格式标准化
- Node.js原生推理
- 性能接近Python

**缺点：**
- 需要转换所有模型
- 某些scikit-learn特性可能不支持
- 增加维护复杂度

#### 选项B: TensorFlow.js
```javascript
// 重新训练模型
const tf = require('@tensorflow/tfjs-node');
const model = await tf.loadLayersModel('file://herg_model.json');
```

**缺点：**
- 需要重新训练
- 随机森林→神经网络（模型类型改变）
- 性能可能下降

#### 选项C: 保留Python子进程
```javascript
// Node.js wrapper调用Python
const { spawn } = require('child_process');

function predictPython(smiles) {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', ['predict.py', smiles]);
    // 处理输出...
  });
}
```

**缺点：**
- 增加延迟
- 进程管理复杂
- 失去Node.js优势

---

## 💡 推荐方案

### 方案1: 混合架构 (推荐) ⭐⭐⭐⭐⭐

**核心思想：** 保持Python科学计算核心，Node.js作为Skill包装层

```
pharma-ai-skill/ (Node.js Skill)
├── package.json
├── src/
│   ├── index.ts              # Skill入口
│   ├── commands/             # 命令处理
│   │   ├── predict.ts        # 预测命令
│   │   ├── train.ts          # 训练命令
│   │   └── screen.ts         # 筛选命令
│   └── python-bridge/        # Python桥接
│       ├── bridge.ts         # 子进程管理
│       └── interfaces.ts     # 类型定义
├── python-core/              # Python核心 (子目录)
│   ├── predict.py
│   ├── train.py
│   ├── models/               # 预训练模型
│   └── requirements.txt
└── SKILL.md
```

**工作流程：**
```
User Request
    ↓
OpenClaw Agent (Node.js)
    ↓
pharma-ai-skill (Node.js)
    ↓ [spawn child process]
Python Core (RDKit + scikit-learn)
    ↓
Return Result
```

**优点：**
- ✅ 保留RDKit完整功能
- ✅ 保留现有模型（无需转换）
- ✅ Node.js Skill原生集成
- ✅ 启动快速（Node.js部分）
- ✅ 开发体验好

**缺点：**
- ⚠️ 子进程通信有轻微延迟 (~100ms)
- ⚠️ 需要维护两种语言

### 方案2: 纯Node.js重构

**适用场景：** 如果愿意牺牲部分功能换取纯Node.js体验

**需要：**
1. 找到RDKit的Node.js替代方案
2. 将所有模型转换为ONNX格式
3. 重新测试验证

**工作量：** 2-3周

**风险：** 高，可能无法完全替代RDKit

### 方案3: 保持现状 (Python Skill)

**现状：** Python代码通过CLI/子进程被OpenClaw调用

**优点：**
- ✅ 无需改造
- ✅ 功能完整

**缺点：**
- ⚠️ 不是"原生"Skill体验
- ⚠️ 启动较慢

---

## 🎯 最终建议

### 推荐: 方案1 (混合架构)

**理由：**

1. **科学计算不可替代**：RDKit是化学信息学的事实标准，没有成熟的Node.js替代
2. **模型资产保护**：现有模型（ROC-AUC 0.852）无需重新训练
3. **最佳用户体验**：Node.js Skill包装提供快速启动和原生集成
4. **合理权衡**：100ms子进程延迟在药物发现场景可接受

### 实施步骤

```bash
# 1. 创建Node.js Skill结构
mkdir pharma-ai-skill
cd pharma-ai-skill
npm init -y
npm install @openclaw/skill-sdk onnxruntime-node

# 2. 复制Python核心
mkdir python-core
cp ~/projects/pharmaai/*.py python-core/
cp -r ~/projects/pharmaai/models python-core/

# 3. 创建Node.js桥接层
mkdir src/python-bridge
cat > src/python-bridge/bridge.ts << 'EOF'
import { spawn } from 'child_process';
import { PythonInput, PythonOutput } from './interfaces';

export async function callPython(
  script: string, 
  input: PythonInput
): Promise<PythonOutput> {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', [
      `${__dirname}/../../python-core/${script}.py`,
      JSON.stringify(input)
    ]);
    
    let output = '';
    python.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(output));
      } else {
        reject(new Error(`Python exit code: ${code}`));
      }
    });
  });
}
EOF

# 4. 创建Skill命令
cat > src/commands/predict.ts << 'EOF'
import { callPython } from '../python-bridge/bridge';

export async function predict(smiles: string) {
  const result = await callPython('predict', { smiles });
  return {
    hERG: result.herg_risk,
    hepatotoxicity: result.hep_risk,
    ames: result.ames_risk,
    overall: result.overall_assessment
  };
}
EOF

# 5. 编写SKILL.md
cat > SKILL.md << 'EOF'
---
name: pharma-ai
description: |
  智能药物发现AI助手，基于RDKit和机器学习模型提供分子毒性预测、
  ADMET评估和虚拟筛选。使用Python科学计算核心，Node.js Skill包装。
  
  Use when: 需要预测分子毒性、评估ADMET性质、筛选候选药物
---

# PharmaAI Skill

## 使用方式

```typescript
import { predict } from './commands/predict';

const result = await predict('CCO');
console.log(result.hERG); // 'Low'
```
EOF
```

---

## 📊 决策矩阵

| 方案 | 开发时间 | 功能完整性 | 性能 | 维护成本 | 推荐度 |
|------|---------|-----------|------|---------|--------|
| 混合架构 | 1-2天 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 纯Node.js | 2-3周 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 保持Python | 0天 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## ✅ 结论

**Node.js更适合OpenClaw集成，但Python科学计算不可替代。**

**最佳方案：混合架构**
- Node.js Skill提供原生集成体验
- Python核心保留RDKit和现有模型
- 子进程桥接实现两者协同

**是否实施混合架构改造？** 我可以立即开始创建Node.js Skill结构。
