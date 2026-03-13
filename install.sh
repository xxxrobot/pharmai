#!/bin/bash
echo "🧪 安装PharmaAI..."
echo ""

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查模型
if [ ! -f "models/herg_model.pkl" ]; then
    echo ""
    echo "⚠️ 模型文件不存在，开始训练..."
    python3 train_all_models.py
fi

echo ""
echo "✅ 安装完成!"
echo ""
echo "🚀 启动方式:"
echo "   python3 launch_web.py"
echo ""
echo "📖 查看文档:"
echo "   PHARMA_WORKFLOW_MANUAL.md"
