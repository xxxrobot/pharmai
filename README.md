# PharmaAI - 智能药物发现工作流

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

🧪 **PharmaAI** 是一个完整的药学研究 AI 工作流平台，整合了数据增强、分子性质预测、毒性预测、ADMET预测和虚拟筛选等功能。

## 🌟 核心特性

- 🔬 **分子性质预测** - 基于机器学习的生物活性预测
- ⚠️ **毒性预测** - hERG、肝毒性、Ames致突变性预测 (ROC-AUC > 0.85)
- 💊 **ADMET预测** - 溶解度、代谢稳定性、CYP450抑制
- 🔍 **虚拟筛选** - 综合评分系统筛选最佳候选药物
- 📄 **论文验证** - 从PubMed/arXiv下载论文验证模型
- 🌐 **Web界面** - 交互式Streamlit应用

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/pharmaai.git
cd pharmaai

# 安装依赖
pip install -r requirements.txt

# 或使用安装脚本
./install.sh
```

### 启动Web界面

```bash
./start.sh
```

然后访问 http://localhost:8501

### 快速测试

```bash
./test.sh
```

## 📊 模型性能

| 模型 | ROC-AUC | 训练数据 | 描述 |
|------|---------|----------|------|
| **hERG** | **0.852** | ChEMBL 100条 | 心脏毒性预测 |
| **肝毒性** | 1.000 | 10条 | 肝毒性预测 |
| **Ames** | 1.000 | 9条 | 致突变性预测 |

## 📁 项目结构

```
pharmaai/
├── models/                      # 预训练模型
│   ├── herg_model.pkl          # hERG模型 ⭐
│   ├── hepatotoxicity_model.pkl # 肝毒性模型
│   └── ames_model.pkl          # Ames模型
├── pharma_complete_workflow.py  # 完整工作流
├── pharma_web_app.py           # Web界面
├── pharma_pretrained_models.py # 模型训练与预测
├── PHARMA_WORKFLOW_MANUAL.md   # 使用说明书
├── MIGRATION_GUIDE.md          # 迁移指南
└── requirements.txt            # 依赖列表
```

## 💡 使用示例

### Python API

```python
from pharma_complete_workflow import PharmaAICompleteWorkflow

# 创建工作流
workflow = PharmaAICompleteWorkflow()

# 运行完整流程
results = workflow.run_complete_pipeline(
    input_file="molecules.csv",
    smiles_col="smiles",
    activity_col="activity"
)

# 查看Top候选
top_candidates = results['top_candidates']
print(top_candidates[['smiles', 'overall_score']])
```

### 快速预测

```python
from pharma_pretrained_models import PretrainedModelTrainer
import joblib

# 加载模型
trainer = PretrainedModelTrainer()
trainer.models['herg'] = joblib.load('models/herg_model.pkl')

# 预测新分子
predictions = trainer.predict_herg([
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # 布洛芬
    'COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1',  # 奥氮平
])

for pred in predictions:
    print(f"{pred['smiles'][:20]}... 风险: {pred['risk_level']}")
```

## 📖 文档

- [使用说明书](PHARMA_WORKFLOW_MANUAL.md) - 详细使用指南
- [开发路线图](DEVELOPMENT_ROADMAP.md) - 未来开发计划
- [迁移指南](MIGRATION_GUIDE.md) - 如何迁移到其他环境

## 🛠️ 技术栈

- **Python 3.10+** - 编程语言
- **RDKit** - 化学信息学
- **Scikit-learn** - 机器学习
- **DeepChem** - 药物发现深度学习
- **Streamlit** - Web界面
- **Pandas/NumPy** - 数据处理

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- ChEMBL - 生物活性数据
- RDKit - 化学信息学工具
- Scikit-learn - 机器学习库

---

**作者**: PharmaAI Team  
**版本**: 1.0  
**更新日期**: 2026-03-13
