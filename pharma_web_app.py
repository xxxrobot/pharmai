#!/usr/bin/env python3
"""
PharmaAI Web 界面 - Streamlit App
药学研究 AI 工作流的可视化界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from io import StringIO, BytesIO
import base64

# 添加工作流路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入工作流模块
try:
    from pharma_complete_workflow import PharmaAICompleteWorkflow, WorkflowConfig
    from pharma_data_enhancement import DataEnhancement
    from pharma_toxicity_prediction import ToxicityPrediction
    from pharma_admet_prediction import ADMETPrediction
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    st.error(f"工作流模块导入失败: {e}")

# RDKit
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski

# 页面配置
st.set_page_config(
    page_title="PharmaAI - 药物发现工作流",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def mol_to_img(smiles, size=(300, 300)):
    """将SMILES转换为图像"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=size)
            return img
    except:
        pass
    return None


def get_risk_color(risk):
    """获取风险等级的颜色"""
    if risk == 'Low' or risk == 'High':
        return 'green'
    elif risk == 'Medium':
        return 'orange'
    else:
        return 'red'


def get_risk_emoji(risk):
    """获取风险等级的表情"""
    if risk == 'Low':
        return '✅'
    elif risk == 'Medium':
        return '⚠️'
    else:
        return '❌'


# ==================== 页面内容 ====================

def main():
    # 标题
    st.markdown('<div class="main-header">🧪 PharmaAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">智能药物发现工作流平台</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("📋 导航菜单")
        
        page = st.radio(
            "选择功能",
            ["🏠 首页", "📊 数据上传", "🔬 分子分析", "⚠️ 毒性预测", 
             "💊 ADMET预测", "🔍 虚拟筛选", "📈 结果展示"]
        )
        
        st.markdown("---")
        st.header("⚙️ 配置")
        
        enable_lipinski = st.checkbox("Lipinski筛选", value=True)
        enable_toxicity = st.checkbox("毒性预测", value=True)
        enable_solubility = st.checkbox("溶解度预测", value=True)
        enable_metabolism = st.checkbox("代谢预测", value=True)
        
        st.markdown("---")
        st.info("💡 提示：上传CSV文件开始分析")
    
    # 页面路由
    if page == "🏠 首页":
        show_home()
    elif page == "📊 数据上传":
        show_data_upload()
    elif page == "🔬 分子分析":
        show_molecule_analysis()
    elif page == "⚠️ 毒性预测":
        show_toxicity_prediction()
    elif page == "💊 ADMET预测":
        show_admet_prediction()
    elif page == "🔍 虚拟筛选":
        show_virtual_screening()
    elif page == "📈 结果展示":
        show_results()


def show_home():
    """首页"""
    st.header("欢迎使用 PharmaAI")
    
    # 功能介绍
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 数据管理
        - 数据清洗与验证
        - SMILES标准化
        - 分子去重
        - 质量评估
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 性质预测
        - 分子描述符计算
        - 生物活性预测
        - Lipinski筛选
        - 毒性风险评估
        """)
    
    with col3:
        st.markdown("""
        ### 💊 ADMET预测
        - 水溶性预测
        - 代谢稳定性
        - CYP450抑制
        - 综合风险评估
        """)
    
    # 统计数据
    st.markdown("---")
    st.subheader("📈 平台统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">29+</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">分子特征</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">6</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">预测模型</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">100%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">开源免费</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">1-click</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">完整流程</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 快速开始
    st.markdown("---")
    st.subheader("🚀 快速开始")
    
    st.markdown("""
    1. **上传数据**: 在"📊 数据上传"页面上传包含SMILES的CSV文件
    2. **分子分析**: 在"🔬 分子分析"页面查看分子性质
    3. **毒性预测**: 在"⚠️ 毒性预测"页面评估安全性
    4. **ADMET预测**: 在"💊 ADMET预测"页面预测药代动力学
    5. **虚拟筛选**: 在"🔍 虚拟筛选"页面筛选最佳候选
    """)
    
    # 示例数据下载
    st.markdown("---")
    st.subheader("📥 示例数据")
    
    sample_data = """smiles,activity,name
CC(C)Cc1ccc(cc1)C(C)C(=O)O,0.85,Ibuprofen
CC(=O)Oc1ccccc1C(=O)O,0.82,Aspirin
CC(C)NCC(COc1ccccc1)O,0.91,Propranolol
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0.75,Caffeine
COc1ccc2nc(N3CCN(C)CC3)nc(C)c2c1,0.88,Olanzapine
"""
    
    st.download_button(
        label="⬇️ 下载示例CSV",
        data=sample_data,
        file_name="sample_molecules.csv",
        mime="text/csv"
    )


def show_data_upload():
    """数据上传页面"""
    st.header("📊 数据上传")
    
    uploaded_file = st.file_uploader(
        "上传分子数据 (CSV格式)",
        type=['csv'],
        help="文件应包含SMILES列"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ 成功加载 {len(df)} 条记录")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head(10))
            
            # 数据验证
            if 'smiles' in df.columns:
                st.markdown('<div class="success-box">✅ 检测到SMILES列</div>', 
                          unsafe_allow_html=True)
                
                # 验证SMILES
                df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) else None)
                valid_count = df['mol'].notna().sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总记录", len(df))
                with col2:
                    st.metric("有效分子", valid_count)
                
                # 保存到session state
                st.session_state['data'] = df
                st.session_state['valid_count'] = valid_count
                
                if valid_count < len(df):
                    st.warning(f"⚠️ {len(df) - valid_count} 个无效SMILES")
            else:
                st.error("❌ 未检测到SMILES列，请确保列名为'smiles'")
                
        except Exception as e:
            st.error(f"❌ 文件读取失败: {e}")
    else:
        st.info("👆 请上传CSV文件开始分析")


def show_molecule_analysis():
    """分子分析页面"""
    st.header("🔬 分子分析")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ 请先上传数据")
        return
    
    df = st.session_state['data']
    
    # 选择分子
    selected_idx = st.selectbox(
        "选择分子",
        range(len(df)),
        format_func=lambda i: f"{i+1}. {df.iloc[i].get('name', 'Unknown')} - {df.iloc[i]['smiles'][:30]}..."
    )
    
    selected_mol = df.iloc[selected_idx]
    smiles = selected_mol['smiles']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 显示分子结构
        st.subheader("分子结构")
        img = mol_to_img(smiles, size=(400, 400))
        if img:
            st.image(img)
        else:
            st.error("无法生成分子图像")
    
    with col2:
        # 计算描述符
        st.subheader("分子描述符")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = {
                '分子量 (MW)': f"{Descriptors.MolWt(mol):.2f}",
                'LogP': f"{Crippen.MolLogP(mol):.2f}",
                'TPSA': f"{Descriptors.TPSA(mol):.2f} Å²",
                '氢键供体': Lipinski.NumHDonors(mol),
                '氢键受体': Lipinski.NumHAcceptors(mol),
                '可旋转键': Lipinski.NumRotatableBonds(mol),
                '芳香环数': Lipinski.NumAromaticRings(mol),
                '重原子数': mol.GetNumHeavyAtoms(),
            }
            
            desc_df = pd.DataFrame(list(descriptors.items()), 
                                  columns=['描述符', '值'])
            st.table(desc_df)
            
            # Lipinski评估
            st.subheader("Lipinski五规则")
            violations = 0
            checks = []
            
            if Descriptors.MolWt(mol) > 500:
                violations += 1
                checks.append(("分子量 ≤ 500", "❌", "red"))
            else:
                checks.append(("分子量 ≤ 500", "✅", "green"))
            
            if Crippen.MolLogP(mol) > 5:
                violations += 1
                checks.append(("LogP ≤ 5", "❌", "red"))
            else:
                checks.append(("LogP ≤ 5", "✅", "green"))
            
            if Lipinski.NumHDonors(mol) > 5:
                violations += 1
                checks.append(("氢键供体 ≤ 5", "❌", "red"))
            else:
                checks.append(("氢键供体 ≤ 5", "✅", "green"))
            
            if Lipinski.NumHAcceptors(mol) > 10:
                violations += 1
                checks.append(("氢键受体 ≤ 10", "❌", "red"))
            else:
                checks.append(("氢键受体 ≤ 10", "✅", "green"))
            
            for check, status, color in checks:
                st.markdown(f"<span style='color:{color}'>{status} {check}</span>", 
                          unsafe_allow_html=True)
            
            if violations <= 1:
                st.markdown('<div class="success-box">✅ 通过Lipinski筛选</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">⚠️ 违反{violations}条规则</div>', 
                          unsafe_allow_html=True)


def show_toxicity_prediction():
    """毒性预测页面"""
    st.header("⚠️ 毒性预测")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ 请先上传数据")
        return
    
    df = st.session_state['data']
    
    if st.button("🔮 运行毒性预测", type="primary"):
        with st.spinner("正在预测毒性..."):
            try:
                # 使用毒性预测模块
                tox = ToxicityPrediction(output_dir="./pharma_web")
                
                # 计算特征
                df = tox.calculate_toxicity_features(df)
                
                # 预测 (简化版，使用规则)
                df['hERG_risk'] = df.apply(lambda row: 
                    'High' if row.get('has_basic_amine', False) and row.get('LogP', 0) > 2 
                    else 'Low', axis=1)
                
                df['hepatotoxic_risk'] = df.apply(lambda row:
                    'High' if row.get('has_nitro', False) or row.get('has_halogenated_aromatic', False)
                    else 'Low', axis=1)
                
                df['ames_risk'] = df.apply(lambda row:
                    'High' if row.get('has_aromatic_amine', False) or row.get('has_nitro', False)
                    else 'Low', axis=1)
                
                # 综合风险
                def overall_risk(row):
                    risks = [row['hERG_risk'], row['hepatotoxic_risk'], row['ames_risk']]
                    if 'High' in risks:
                        return 'High'
                    elif 'Medium' in risks:
                        return 'Medium'
                    return 'Low'
                
                df['overall_toxicity_risk'] = df.apply(overall_risk, axis=1)
                
                st.session_state['data'] = df
                st.session_state['toxicity_done'] = True
                
                st.success("✅ 毒性预测完成!")
                
            except Exception as e:
                st.error(f"❌ 预测失败: {e}")
    
    if st.session_state.get('toxicity_done'):
        df = st.session_state['data']
        
        # 显示结果
        st.subheader("预测结果")
        
        display_cols = ['smiles', 'hERG_risk', 'hepatotoxic_risk', 'ames_risk', 'overall_toxicity_risk']
        available_cols = [c for c in display_cols if c in df.columns]
        
        if available_cols:
            result_df = df[available_cols].copy()
            
            # 添加表情
            for col in ['hERG_risk', 'hepatotoxic_risk', 'ames_risk', 'overall_toxicity_risk']:
                if col in result_df.columns:
                    result_df[col] = result_df[col].apply(
                        lambda x: f"{get_risk_emoji(x)} {x}"
                    )
            
            st.dataframe(result_df)
            
            # 统计
            st.subheader("风险统计")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'hERG_risk' in df.columns:
                    st.write("**hERG毒性**")
                    st.write(df['hERG_risk'].value_counts())
            
            with col2:
                if 'hepatotoxic_risk' in df.columns:
                    st.write("**肝毒性**")
                    st.write(df['hepatotoxic_risk'].value_counts())
            
            with col3:
                if 'ames_risk' in df.columns:
                    st.write("**致突变性**")
                    st.write(df['ames_risk'].value_counts())


def show_admet_prediction():
    """ADMET预测页面"""
    st.header("💊 ADMET预测")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ 请先上传数据")
        return
    
    df = st.session_state['data']
    
    if st.button("🔮 运行ADMET预测", type="primary"):
        with st.spinner("正在预测ADMET性质..."):
            try:
                admet = ADMETPrediction(output_dir="./pharma_web")
                
                # 计算特征
                df = admet.calculate_solubility_features(df)
                df = admet.calculate_metabolism_features(df)
                
                # 溶解度预测
                def solubility_class(row):
                    ratio = row.get('TPSA_ratio', 0)
                    if ratio > 0.25:
                        return 'High'
                    elif ratio > 0.15:
                        return 'Medium'
                    return 'Low'
                
                df['solubility_class'] = df.apply(solubility_class, axis=1)
                
                # 代谢稳定性预测
                df['metabolic_stability'] = df.apply(lambda row:
                    'High' if row.get('NumMetabolicSites', 0) < 3
                    else 'Medium' if row.get('NumMetabolicSites', 0) < 6
                    else 'Low', axis=1)
                
                st.session_state['data'] = df
                st.session_state['admet_done'] = True
                
                st.success("✅ ADMET预测完成!")
                
            except Exception as e:
                st.error(f"❌ 预测失败: {e}")
    
    if st.session_state.get('admet_done'):
        df = st.session_state['data']
        
        st.subheader("ADMET预测结果")
        
        display_cols = ['smiles', 'solubility_class', 'metabolic_stability']
        available_cols = [c for c in display_cols if c in df.columns]
        
        if available_cols:
            st.dataframe(df[available_cols])


def show_virtual_screening():
    """虚拟筛选页面"""
    st.header("🔍 虚拟筛选")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ 请先上传数据")
        return
    
    df = st.session_state['data']
    
    top_n = st.slider("选择Top N候选", 1, min(50, len(df)), 5)
    
    if st.button("🔍 运行虚拟筛选", type="primary"):
        with st.spinner("正在筛选..."):
            try:
                # 计算综合评分
                scores = np.zeros(len(df))
                
                # 如果有活性预测
                if 'activity' in df.columns:
                    scores += df['activity'] * 0.4
                
                # 毒性惩罚
                if 'overall_toxicity_risk' in df.columns:
                    penalty = df['overall_toxicity_risk'].map({'Low': 0, 'Medium': -0.2, 'High': -0.5})
                    scores += penalty.fillna(0)
                
                # 溶解度奖励
                if 'solubility_class' in df.columns:
                    bonus = df['solubility_class'].map({'Low': -0.1, 'Medium': 0, 'High': 0.1})
                    scores += bonus.fillna(0)
                
                df['overall_score'] = scores
                
                # 排序
                top_candidates = df.nlargest(top_n, 'overall_score')
                
                st.session_state['top_candidates'] = top_candidates
                st.session_state['screening_done'] = True
                
                st.success(f"✅ 筛选完成! 找到 {len(top_candidates)} 个候选")
                
            except Exception as e:
                st.error(f"❌ 筛选失败: {e}")
    
    if st.session_state.get('screening_done'):
        top_candidates = st.session_state['top_candidates']
        
        st.subheader(f"Top {len(top_candidates)} 候选药物")
        
        for idx, (_, row) in enumerate(top_candidates.iterrows()):
            with st.expander(f"#{idx+1} - 综合评分: {row.get('overall_score', 0):.3f}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    img = mol_to_img(row['smiles'], size=(300, 300))
                    if img:
                        st.image(img)
                
                with col2:
                    st.write(f"**SMILES**: `{row['smiles']}`")
                    
                    if 'activity' in row:
                        st.write(f"**活性**: {row['activity']:.3f}")
                    
                    if 'overall_toxicity_risk' in row:
                        emoji = get_risk_emoji(row['overall_toxicity_risk'])
                        st.write(f"**毒性风险**: {emoji} {row['overall_toxicity_risk']}")
                    
                    if 'solubility_class' in row:
                        st.write(f"**溶解度**: {row['solubility_class']}")


def show_results():
    """结果展示页面"""
    st.header("📈 结果展示")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ 请先上传数据并运行分析")
        return
    
    df = st.session_state['data']
    
    # 下载结果
    st.subheader("📥 下载结果")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ 下载完整结果 (CSV)",
        data=csv,
        file_name="pharmaai_results.csv",
        mime="text/csv"
    )
    
    # 数据分布图
    st.subheader("📊 数据分布")
    
    if 'MW' in df.columns:
        st.write("**分子量分布**")
        st.bar_chart(df['MW'])
    
    if 'LogP' in df.columns:
        st.write("**LogP分布**")
        st.bar_chart(df['LogP'])
    
    # 风险分布
    if 'overall_toxicity_risk' in df.columns:
        st.write("**毒性风险分布**")
        risk_counts = df['overall_toxicity_risk'].value_counts()
        st.bar_chart(risk_counts)


# ==================== 主入口 ====================

if __name__ == "__main__":
    main()
