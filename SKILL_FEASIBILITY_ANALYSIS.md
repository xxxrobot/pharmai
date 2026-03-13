# PharmaAI 改造为 OpenClaw Agent Skill 可行性分析

## 📊 当前状态分析

### PharmaAI 现有架构

```
PharmaAI v1.0
├── 核心模块 (9个Python文件)
│   ├── pharma_complete_workflow.py    # 主工作流
│   ├── pharma_web_app.py             # Streamlit Web界面
│   ├── pharma_pretrained_models.py   # 预训练模型
│   ├── train_all_models.py           # 模型训练
│   ├── pharma_data_enhancement.py    # 数据增强
│   ├── pharma_toxicity_prediction.py # 毒性预测
│   ├── pharma_admet_prediction.py    # ADMET预测
│   ├── pharma_paper_validation.py    # 论文验证
│   └── launch_web.py                 # 启动脚本
├── 模型文件 (3个.pkl文件, ~500KB)
│   ├── herg_model.pkl
│   ├── hepatotoxicity_model.pkl
│   └── ames_model.pkl
├── 文档 (4个Markdown文件)
└── 依赖: RDKit, scikit-learn, pandas, streamlit
```

### OpenClaw Skill 要求

```
Skill 标准结构
├── SKILL.md (必需)          # YAML frontmatter + 指令
├── scripts/ (可选)          # 可执行脚本
├── references/ (可选)       # 参考文档
└── assets/ (可选)           # 资源文件
```

---

## ✅ 改造可行性: HIGH (高度可行)

### 优势分析

| 方面 | PharmaAI现状 | Skill适配性 | 评估 |
|------|-------------|------------|------|
| **模块化** | 9个独立模块 | 可映射到scripts/ | ✅ 优秀 |
| **确定性** | ML模型预测 | 确定性输出 | ✅ 良好 |
| **可复用** | 完整工作流 | 可封装为技能 | ✅ 优秀 |
| **文档** | 已有完整文档 | 可整理为SKILL.md | ✅ 良好 |
| **依赖** | RDKit等科学包 | OpenClaw支持 | ✅ 可行 |

### 核心改造点

#### 1. 架构映射

```
PharmaAI → Skill

pharma_complete_workflow.py  →  SKILL.md (主指令)
pharma_pretrained_models.py  →  scripts/predict.py
pharma_data_enhancement.py   →  scripts/data_utils.py
models/*.pkl                 →  assets/models/
PHARMA_WORKFLOW_MANUAL.md    →  references/manual.md
```

#### 2. 交互方式改造

| 当前 | Skill改造后 |
|------|------------|
| Streamlit Web界面 | Agent对话式交互 |
| 文件上传CSV | Agent读取工作区文件 |
| 可视化展示 | Agent生成报告/图表 |
| 批量处理 | Agent后台执行 |

---

## 🎯 推荐改造方案

### 方案A: 完整Skill (推荐)

**复杂度**: ⭐⭐⭐  
**价值**: ⭐⭐⭐⭐⭐

包含完整功能：数据清洗 → 特征计算 → 预测 → 报告

```
pharma-ai/
├── SKILL.md
├── scripts/
│   ├── predict.py          # 核心预测脚本
│   ├── train_model.py      # 模型训练
│   ├── data_cleaner.py     # 数据清洗
│   └── validate.py         # 验证工具
├── references/
│   ├── workflow.md         # 工作流详细说明
│   ├── model_guide.md      # 模型使用指南
│   └── chembl_api.md       # ChEMBL API文档
└── assets/
    └── models/             # 预训练模型
        ├── herg_model.pkl
        ├── hepatotoxicity_model.pkl
        └── ames_model.pkl
```

**SKILL.md 示例**:
```yaml
---
name: pharma-ai
description: |
  智能药物发现AI助手，提供分子性质预测、毒性评估、ADMET预测和虚拟筛选功能。
  
  Use when:
  - 需要预测分子的hERG心脏毒性、肝毒性或Ames致突变性
  - 需要评估分子的溶解度、代谢稳定性等ADMET性质
  - 需要从化合物库中筛选候选药物
  - 需要验证分子是否符合Lipinski五规则
  - 需要下载ChEMBL数据训练新模型
  
  Supports: CSV/SDF分子数据, ChEMBL API, 批量预测
---

# PharmaAI Skill

## 快速开始

预测单个分子:
```python
from scripts.predict import predict_toxicity
result = predict_toxicity('CCO')  # 乙醇
print(result['hERG_risk'])  # Low
```

批量预测:
```python
from scripts.predict import batch_predict
results = batch_predict('molecules.csv')
```

## 模型性能
- hERG: ROC-AUC 0.852
- 肝毒性: ROC-AUC 1.000
- Ames: ROC-AUC 1.000
```

### 方案B: 轻量Skill

**复杂度**: ⭐⭐  
**价值**: ⭐⭐⭐⭐

只保留核心预测功能，模型训练作为reference文档

```
pharma-predictor/
├── SKILL.md
├── scripts/
│   └── predict.py          # 仅预测功能
└── assets/
    └── models/             # 预训练模型
```

### 方案C: 工具集Skill

**复杂度**: ⭐⭐⭐⭐  
**价值**: ⭐⭐⭐⭐⭐

将功能拆分为多个小skill：

```
pharma-toxicity-predictor/     # 毒性预测
pharma-admet-predictor/        # ADMET预测  
pharma-virtual-screener/       # 虚拟筛选
pharma-chembl-connector/       # ChEMBL数据
```

---

## 🔧 技术实现细节

### 依赖处理

```python
# requirements.txt (Skill级别)
rdkit>=2023.0.0      # 必需: 化学信息学
scikit-learn>=1.3.0  # 必需: 机器学习
pandas>=1.5.0        # 必需: 数据处理
numpy>=1.24.0        # 必需: 数值计算
joblib>=1.3.0        # 必需: 模型加载
```

OpenClaw环境已支持这些包 ✅

### 模型文件处理

```python
# scripts/predict.py
import os
import joblib

# 获取skill目录
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SKILL_DIR, 'assets', 'models')

def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
    return joblib.load(model_path)
```

### 输入输出适配

```python
# 适配Agent对话式交互
def predict_molecule(smiles: str) -> dict:
    """
    预测单个分子
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        {
            'smiles': 'CCO',
            'hERG_risk': 'Low',
            'hERG_prob': 0.05,
            'hepatotoxicity_risk': 'Low',
            'ames_risk': 'Low',
            'solubility': 'High',
            'overall_assessment': 'Safe'
        }
    """
    # 实现预测逻辑
    pass
```

---

## 📈 改造工作量估算

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 创建SKILL.md | 2-3小时 | 编写frontmatter和核心指令 |
| 重构scripts/ | 4-6小时 | 将模块改为独立脚本 |
| 整理references/ | 2-3小时 | 提取和整理参考文档 |
| 测试验证 | 2-3小时 | 确保所有功能正常 |
| 打包发布 | 1小时 | 使用package_skill.py |
| **总计** | **11-16小时** | 约1.5-2个工作日 |

---

## 🎁 改造后价值

### 对用户的价值

1. **零配置使用**: 无需安装，直接通过Agent调用
2. **对话式交互**: 自然语言描述需求
3. **集成工作流**: 与其他skill协同工作
4. **自动文档**: Agent自动生成使用说明

### 示例使用场景

```
User: 帮我预测这个分子的毒性: CC(C)Cc1ccc(cc1)C(C)C(=O)O

Agent: 我来帮您预测布洛芬的毒性...

🧪 预测结果:
- hERG心脏毒性: Low (概率: 0%)
- 肝毒性: Low (概率: 10%)
- Ames致突变性: Low (概率: 5%)
- 综合评估: 安全

📊 详细报告已保存到: ./toxicity_report.md
```

```
User: 从ChEMBL下载hERG数据训练新模型

Agent: 正在从ChEMBL获取数据...
✅ 获取100条记录
✅ 训练完成，ROC-AUC: 0.87
✅ 新模型已保存
```

---

## ✅ 结论

### 可行性: **HIGH** ✅

PharmaAI非常适合改造为OpenClaw Agent Skill：

1. **架构匹配**: 模块化设计天然适配skill结构
2. **功能确定**: ML模型提供确定性输出
3. **价值明确**: 药物发现是Agent的高价值应用场景
4. **技术可行**: 依赖包OpenClaw都已支持

### 推荐方案: **方案A (完整Skill)**

保留完整功能，提供最大价值。

### 下一步行动

1. ✅ 创建skill目录结构
2. ✅ 编写SKILL.md
3. ✅ 重构核心脚本
4. ✅ 测试验证
5. ✅ 打包发布

**预计完成时间**: 1.5-2个工作日

---

**是否开始改造实施？** 我可以立即开始创建skill结构和编写SKILL.md。
