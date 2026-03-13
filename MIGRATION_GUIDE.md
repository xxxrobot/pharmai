# PharmaAI 项目迁移指南

## 📦 项目打包与迁移

### 方式一：完整项目打包（推荐）

#### 1. 创建项目包

```bash
# 进入工作目录
cd /home/lutao/.openclaw/workspace

# 创建项目目录
mkdir -p pharmaai_project
cd pharmaai_project

# 复制核心文件
cp ../pharma_complete_workflow.py .
cp ../pharma_web_app.py .
cp ../pharma_pretrained_models.py .
cp ../train_all_models.py .
cp ../pharma_data_enhancement.py .
cp ../pharma_toxicity_prediction.py .
cp ../pharma_admet_prediction.py .
cp ../pharma_paper_validation.py .
cp ../launch_web.py .

# 复制文档
cp ../PHARMA_WORKFLOW_MANUAL.md .
cp ../DEVELOPMENT_ROADMAP.md .
cp ../TODO.md .

# 复制模型
mkdir -p models
cp -r ../pharma_models/models/* models/

# 创建requirements.txt
cat > requirements.txt << 'EOF'
rdkit>=2023.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
deepchem>=2.7.0
streamlit>=1.28.0
requests>=2.31.0
EOF

# 创建README.md
cat > README.md << 'EOF'
# PharmaAI - 智能药物发现工作流

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 启动Web界面
```bash
python launch_web.py
```

3. 访问 http://localhost:8501

## 文件说明
- pharma_complete_workflow.py - 完整工作流
- pharma_web_app.py - Web界面
- pharma_pretrained_models.py - 预训练模型
- models/ - 训练好的模型文件
EOF

# 打包
cd ..
tar -czvf pharmaai_v1.0.tar.gz pharmaai_project/
```

#### 2. 在其他agents中部署

```bash
# 解压项目
tar -xzvf pharmaai_v1.0.tar.gz
cd pharmaai_project

# 安装依赖
pip install -r requirements.txt

# 启动
python launch_web.py
```

---

### 方式二：Git仓库迁移

#### 1. 初始化Git仓库

```bash
cd /home/lutao/.openclaw/workspace/pharmaai_project

git init
git add .
git commit -m "PharmaAI v1.0 - 智能药物发现工作流"

# 创建GitHub/GitLab仓库并推送
git remote add origin https://github.com/yourusername/pharmaai.git
git push -u origin master
```

#### 2. 在其他agents中克隆

```bash
git clone https://github.com/yourusername/pharmaai.git
cd pharmaai
pip install -r requirements.txt
python launch_web.py
```

---

### 方式三：Docker容器化

#### 1. 创建Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenbabel-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "pharma_web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. 构建和运行

```bash
# 构建镜像
docker build -t pharmaai:v1.0 .

# 运行容器
docker run -p 8501:8501 pharmaai:v1.0

# 保存镜像
docker save pharmaai:v1.0 > pharmaai_docker.tar

# 在其他机器加载
docker load < pharmaai_docker.tar
docker run -p 8501:8501 pharmaai:v1.0
```

---

## 🔧 迁移到其他Agents的具体步骤

### 步骤1：导出项目

```bash
# 在当前环境执行
cd /home/lutao/.openclaw/workspace

# 创建导出脚本
cat > export_pharmaai.sh << 'SCRIPT'
#!/bin/bash
EXPORT_DIR="pharmaai_export_$(date +%Y%m%d)"
mkdir -p $EXPORT_DIR

echo "导出PharmaAI项目..."

# 核心代码
cp pharma_complete_workflow.py $EXPORT_DIR/
cp pharma_web_app.py $EXPORT_DIR/
cp pharma_pretrained_models.py $EXPORT_DIR/
cp train_all_models.py $EXPORT_DIR/
cp pharma_data_enhancement.py $EXPORT_DIR/
cp pharma_toxicity_prediction.py $EXPORT_DIR/
cp pharma_admet_prediction.py $EXPORT_DIR/
cp pharma_paper_validation.py $EXPORT_DIR/
cp launch_web.py $EXPORT_DIR/

# 文档
cp PHARMA_WORKFLOW_MANUAL.md $EXPORT_DIR/
cp DEVELOPMENT_ROADMAP.md $EXPORT_DIR/
cp TODO.md $EXPORT_DIR/

# 模型
mkdir -p $EXPORT_DIR/models
cp pharma_models/models/*.pkl $EXPORT_DIR/models/ 2>/dev/null || echo "模型文件需要重新训练"

# 依赖
cat > $EXPORT_DIR/requirements.txt << 'EOF'
rdkit>=2023.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
deepchem>=2.7.0
streamlit>=1.28.0
requests>=2.31.0
EOF

# 安装脚本
cat > $EXPORT_DIR/install.sh << 'EOF'
#!/bin/bash
echo "安装PharmaAI..."
pip install -r requirements.txt
echo "安装完成!"
echo "启动: python launch_web.py"
EOF
chmod +x $EXPORT_DIR/install.sh

# 打包
tar -czvf ${EXPORT_DIR}.tar.gz $EXPORT_DIR
echo "导出完成: ${EXPORT_DIR}.tar.gz"
SCRIPT

chmod +x export_pharmaai.sh
./export_pharmaai.sh
```

### 步骤2：传输到其他Agents

```bash
# 方式1：直接复制（如果在同一网络）
scp pharmaai_export_*.tar.gz user@other-agent:/path/to/destination/

# 方式2：上传到云存储
# 上传到Google Drive, Dropbox, AWS S3等

# 方式3：通过共享目录
# 复制到共享文件夹
```

### 步骤3：在其他Agents中导入

```bash
# 解压
tar -xzvf pharmaai_export_20260313.tar.gz
cd pharmaai_export_20260313

# 安装依赖
./install.sh
# 或
pip install -r requirements.txt

# 测试运行
python -c "from pharma_complete_workflow import PharmaAICompleteWorkflow; print('✅ 导入成功')"

# 启动Web界面
python launch_web.py
```

---

## 📋 迁移检查清单

### 环境检查
- [ ] Python 3.10+
- [ ] pip可用
- [ ] 网络连接（下载依赖）
- [ ] 至少4GB内存
- [ ] 至少1GB磁盘空间

### 依赖安装
```bash
# 验证安装
python -c "import rdkit; print('RDKit:', rdkit.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

### 功能测试
```bash
# 测试1：基础导入
python -c "from pharma_complete_workflow import PharmaAICompleteWorkflow"

# 测试2：模型加载
python -c "
import joblib
model = joblib.load('models/herg_model.pkl')
print('✅ 模型加载成功')
"

# 测试3：简单预测
python -c "
from pharma_pretrained_models import PretrainedModelTrainer
trainer = PretrainedModelTrainer()
trainer.models['herg'] = joblib.load('models/herg_model.pkl')
result = trainer.predict_herg(['CCO'])
print('✅ 预测成功:', result)
"

# 测试4：Web界面
python launch_web.py &
sleep 5
curl http://localhost:8501 > /dev/null && echo "✅ Web界面正常"
```

---

## 🔄 同步更新策略

### 方式1：Git同步（推荐）

```bash
# 主仓库更新
git add .
git commit -m "更新功能"
git push origin master

# 其他agents拉取更新
git pull origin master
```

### 方式2：模型文件单独同步

```bash
# 只同步模型文件（大文件）
rsync -avz models/ user@other-agent:/path/to/pharmaai/models/

# 同步代码（小文件）
rsync -avz *.py *.md user@other-agent:/path/to/pharmaai/
```

### 方式3：版本化发布

```bash
# 创建版本标签
git tag -a v1.0 -m "PharmaAI v1.0 发布"
git push origin v1.0

# 其他agents切换到特定版本
git checkout v1.0
```

---

## 🛠️ 常见问题解决

### 问题1：RDKit安装失败
```bash
# 解决方案1：使用conda
conda install -c conda-forge rdkit

# 解决方案2：使用预编译包
pip install rdkit-pypi
```

### 问题2：模型文件缺失
```bash
# 重新训练模型
cd pharmaai_project
python train_all_models.py
```

### 问题3：端口冲突
```bash
# 使用其他端口
streamlit run pharma_web_app.py --server.port 8502
```

### 问题4：内存不足
```bash
# 减少批量处理大小
# 修改配置文件中的 batch_size 参数
```

---

## 📦 一键迁移脚本

创建 `migrate.sh`：

```bash
#!/bin/bash
# PharmaAI 一键迁移脚本

set -e

echo "🧪 PharmaAI 项目迁移工具"
echo "=========================="

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <目标目录>"
    echo "示例: $0 /home/newuser/pharmaai"
    exit 1
fi

TARGET_DIR=$1
SOURCE_DIR="/home/lutao/.openclaw/workspace"

echo ""
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo ""

# 创建目标目录
mkdir -p $TARGET_DIR
cd $TARGET_DIR

# 复制核心文件
echo "📦 复制核心文件..."
cp $SOURCE_DIR/pharma_complete_workflow.py .
cp $SOURCE_DIR/pharma_web_app.py .
cp $SOURCE_DIR/pharma_pretrained_models.py .
cp $SOURCE_DIR/train_all_models.py .
cp $SOURCE_DIR/pharma_data_enhancement.py .
cp $SOURCE_DIR/pharma_toxicity_prediction.py .
cp $SOURCE_DIR/pharma_admet_prediction.py .
cp $SOURCE_DIR/pharma_paper_validation.py .
cp $SOURCE_DIR/launch_web.py .

# 复制文档
echo "📄 复制文档..."
cp $SOURCE_DIR/PHARMA_WORKFLOW_MANUAL.md .
cp $SOURCE_DIR/DEVELOPMENT_ROADMAP.md .
cp $SOURCE_DIR/TODO.md .

# 复制模型
echo "🤖 复制模型文件..."
mkdir -p models
if [ -d "$SOURCE_DIR/pharma_models/models" ]; then
    cp $SOURCE_DIR/pharma_models/models/*.pkl models/ 2>/dev/null || echo "⚠️ 模型文件不存在，需要重新训练"
fi

# 创建requirements.txt
echo "📝 创建依赖文件..."
cat > requirements.txt << 'EOF'
rdkit>=2023.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
deepchem>=2.7.0
streamlit>=1.28.0
requests>=2.31.0
EOF

# 创建启动脚本
echo "🚀 创建启动脚本..."
cat > start.sh << 'EOF'
#!/bin/bash
echo "🧪 启动 PharmaAI..."
python3 launch_web.py
EOF
chmod +x start.sh

# 创建测试脚本
cat > test.sh << 'EOF'
#!/bin/bash
echo "🧪 测试 PharmaAI..."
python3 -c "from pharma_complete_workflow import PharmaAICompleteWorkflow; print('✅ 核心模块导入成功')"
python3 -c "import joblib; model = joblib.load('models/herg_model.pkl'); print('✅ 模型加载成功')" 2>/dev/null || echo "⚠️ 模型文件不存在"
echo "✅ 测试完成"
EOF
chmod +x test.sh

echo ""
echo "✅ 迁移完成!"
echo ""
echo "📂 项目位置: $TARGET_DIR"
echo ""
echo "🔧 下一步:"
echo "   1. cd $TARGET_DIR"
echo "   2. pip install -r requirements.txt"
echo "   3. ./test.sh"
echo "   4. ./start.sh"
echo ""
```

使用方法：
```bash
chmod +x migrate.sh
./migrate.sh /path/to/new/location
```

---

## 📞 支持

如有迁移问题，请检查：
1. Python版本 >= 3.10
2. 所有依赖已安装
3. 模型文件存在或可重新训练
4. 端口未被占用

---

**迁移指南完成！** 选择适合你的方式开始迁移吧。
