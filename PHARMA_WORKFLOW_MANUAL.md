# PharmaAI 工作流使用说明书 v1.0

## 📖 概述

PharmaAI 是一个完整的药学研究 AI 工作流，整合了数据增强、分子性质预测、毒性预测、ADMET预测和虚拟筛选等功能。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install rdkit pandas numpy scikit-learn deepchem
```

### 2. 基本使用

```python
from pharma_complete_workflow import PharmaAICompleteWorkflow, WorkflowConfig

# 创建工作流
workflow = PharmaAICompleteWorkflow()

# 运行完整流程
results = workflow.run_complete_pipeline(
    input_file="your_molecules.csv",
    smiles_col="smiles",
    activity_col="activity"  # 可选
)
```

### 3. 快速预测

```python
# 对新分子进行快速ADMET预测
smiles_list = [
    'CC(C)Cc1ccc(cc1)C(=O)O',  # 布洛芬
    'CC(=O)Oc1ccccc1C(=O)O',   # 阿司匹林
]

predictions = workflow.quick_predict(smiles_list)
print(predictions)
```

## 📁 输入数据格式

### CSV 格式示例

```csv
smiles,activity
CC(C)Cc1ccc(cc1)C(=O)O,0.85
CC(=O)Oc1ccccc1C(=O)O,0.82
CC(C)NCC(COc1ccccc1)O,0.91
```

### 必需列
- `smiles`: SMILES格式的分子结构

### 可选列
- `activity`: 生物活性值 (0-1)
- `name`: 分子名称
- `id`: 分子ID

## ⚙️ 配置选项

```python
config = WorkflowConfig(
    output_dir="./my_results",          # 输出目录
    enable_data_cleaning=True,          # 启用数据清洗
    enable_lipinski_filter=True,        # 启用Lipinski筛选
    enable_toxicity=True,               # 启用毒性预测
    enable_solubility=True,             # 启用溶解度预测
    enable_metabolism=True,             # 启用代谢预测
    enable_cyp=True,                    # 启用CYP预测
    virtual_screening_top_n=100         # 虚拟筛选Top N
)

workflow = PharmaAICompleteWorkflow(config)
```

## 📊 输出结果

### 目录结构

```
pharma_complete/
├── data/                       # 数据文件
│   └── sample_input.csv
├── models/                     # 训练好的模型
│   └── activity_model.pkl
├── results/                    # 预测结果
│   ├── top_candidates.csv      # Top候选药物
│   └── complete_predictions.csv # 完整预测
├── visualizations/             # 可视化
│   └── candidates.png          # 分子结构图
└── reports/                    # 报告
    └── comprehensive_report.json # 综合报告
```

### 输出文件说明

#### 1. complete_predictions.csv
包含所有分子的完整预测结果：
- `smiles`: 分子SMILES
- `MW`: 分子量
- `LogP`: 脂水分配系数
- `TPSA`: 拓扑极性表面积
- `activity_predicted`: 预测活性
- `toxicity_risk`: 毒性风险 (Low/Medium/High)
- `solubility_class`: 溶解度等级 (Low/Medium/High)
- `overall_score`: 综合评分

#### 2. top_candidates.csv
综合评分最高的候选药物

#### 3. comprehensive_report.json
JSON格式的完整报告，包含：
- 工作流信息
- 数据统计
- 模型性能
- 属性分布
- Top候选列表

## 🔬 功能模块

### 1. 数据增强与验证
- 数据清洗和去重
- SMILES标准化
- 数据质量验证
- 缺失值处理

### 2. 分子性质预测
- 分子描述符计算
- 生物活性预测
- Lipinski五规则筛选

### 3. 毒性预测
- hERG心脏毒性
- 肝毒性
- Ames致突变性
- 综合毒性风险评估

### 4. ADMET预测
- 水溶性预测
- 代谢稳定性预测
- CYP450抑制预测
- 药物相互作用风险评估

### 5. 虚拟筛选
- 基于多属性的综合评分
- Top候选药物筛选
- 分子结构可视化

## 💡 使用示例

### 示例1: 基础使用

```python
from pharma_complete_workflow import PharmaAICompleteWorkflow

# 最简单用法
workflow = PharmaAICompleteWorkflow()
results = workflow.run_complete_pipeline("my_data.csv")

# 查看Top候选
top_candidates = results['top_candidates']
print(top_candidates[['smiles', 'overall_score']])
```

### 示例2: 自定义配置

```python
from pharma_complete_workflow import PharmaAICompleteWorkflow, WorkflowConfig

config = WorkflowConfig(
    output_dir="./my_project",
    virtual_screening_top_n=50,
    enable_lipinski_filter=True
)

workflow = PharmaAICompleteWorkflow(config)
results = workflow.run_complete_pipeline(
    "my_data.csv",
    smiles_col="SMILES",
    activity_col="IC50"
)
```

### 示例3: 批量预测

```python
# 预测新化合物库
import pandas as pd

# 加载化合物库
library = pd.read_csv("compound_library.csv")

# 快速预测
predictions = workflow.quick_predict(library['smiles'].tolist())

# 筛选低风险化合物
safe_compounds = predictions[predictions['toxicity_risk'] == 'Low']
print(f"安全化合物: {len(safe_compounds)}")
```

## 📈 模型性能

### 当前模型性能

| 模型 | 类型 | 性能指标 | 说明 |
|------|------|---------|------|
| 活性预测 | 回归 | R² (取决于数据) | 基于Morgan指纹的随机森林 |
| 毒性预测 | 分类 | 规则-based | 基于警示结构 |
| 溶解度 | 分类 | 规则-based | 基于TPSA/MW比值 |
| 代谢稳定性 | 回归 | 需更多数据 | 基于代谢位点 |

## 🔧 故障排除

### 问题1: RDKit导入错误
```bash
# 解决方案
conda install -c conda-forge rdkit
# 或
pip install rdkit-pypi
```

### 问题2: 内存不足
```python
# 分批处理
config = WorkflowConfig(
    virtual_screening_top_n=50  # 减少Top N
)
```

### 问题3: 模型性能差
- 增加训练数据量
- 检查数据质量
- 考虑使用预训练模型

## 📚 API参考

### PharmaAICompleteWorkflow 类

#### 主要方法

**`run_complete_pipeline(input_file, smiles_col='smiles', activity_col=None)`**
- 运行完整工作流
- 返回包含所有结果的字典

**`quick_predict(smiles_list)`**
- 快速预测新分子
- 返回DataFrame格式的预测结果

**`load_data(file_path, smiles_col, activity_col)`**
- 加载数据文件

**`clean_data(df)`**
- 清洗数据

**`virtual_screening(df, top_n)`**
- 虚拟筛选

## 📞 支持

如有问题，请查看：
1. 工作流日志: `pharma_complete/workflow.log`
2. 示例代码: `pharma_complete_workflow.py`
3. 本说明书

## 📄 许可证

MIT License

## 🙏 致谢

- RDKit: 化学信息学工具包
- Scikit-learn: 机器学习库
- DeepChem: 药物发现深度学习

---

**版本**: 1.0  
**更新日期**: 2026-03-13
