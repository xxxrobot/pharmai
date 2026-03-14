"""
CYP450预测模块
基于现有毒性预测框架扩展CYP450抑制预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

class CYP450Predictor:
    """CYP450抑制预测器"""
    
    def __init__(self, cyp_isoform='CYP3A4'):
        """
        初始化CYP450预测器
        
        Args:
            cyp_isoform: CYP亚型 ('CYP3A4', 'CYP2D6', 'CYP2C9')
        """
        self.cyp_isoform = cyp_isoform
        self.model = None
        self.desc_cols = None
        self.feature_importance = None
        
    def calculate_features(self, smiles):
        """
        计算分子特征
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            dict: 特征字典
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # 基础描述符 (与现有毒性预测保持一致)
        features = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotatableBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumNitrogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
            'NumHalogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]),
            'NumSulfurs': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
            'NumOxygens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        }
        
        # Morgan指纹 (2048位，与现有模型保持一致)
        fp = GetMorganFingerprintAsBitVect(mol, 2, 2048)
        features['fingerprint'] = np.array(fp)
        
        return features
    
    def prepare_features(self, df):
        """
        准备特征矩阵
        
        Args:
            df: 包含'smiles'和'is_inhibitor'列的DataFrame
            
        Returns:
            tuple: (X, y) 特征矩阵和标签
        """
        X_desc_list = []
        X_fp_list = []
        y_list = []
        
        desc_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 
                    'AromaticRings', 'HeavyAtoms', 'NumNitrogens', 
                    'NumHalogens', 'NumSulfurs', 'NumOxygens']
        
        for idx, row in df.iterrows():
            features = self.calculate_features(row['smiles'])
            if features:
                X_desc = [features[col] for col in desc_cols]
                X_desc_list.append(X_desc)
                X_fp_list.append(features['fingerprint'])
                y_list.append(row['is_inhibitor'])
        
        if not X_desc_list:
            return None, None, None
        
        X_desc_arr = np.array(X_desc_list)
        X_fp_arr = np.array(X_fp_list)
        X = np.hstack([X_desc_arr, X_fp_arr])
        y = np.array(y_list)
        
        self.desc_cols = desc_cols
        return X, y, desc_cols
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        训练CYP450预测模型
        
        Args:
            df: 训练数据
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            dict: 训练结果
        """
        print(f"训练 {self.cyp_isoform} 预测模型...")
        
        # 准备特征
        X, y, desc_cols = self.prepare_features(df)
        if X is None or len(X) == 0:
            return {'error': '无法从数据中提取特征'}
        
        print(f"  样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}")
        print(f"  抑制剂比例: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算AUC（需要至少两个类别）
        if len(np.unique(y_test)) > 1:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            auc_score = None
        
        # 特征重要性
        self.feature_importance = self.model.feature_importances_
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        # 构建结果
        results = {
            'cyp_isoform': self.cyp_isoform,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'auc_score': auc_score,
            'cv_auc_mean': cv_scores.mean() if len(cv_scores) > 0 else None,
            'cv_auc_std': cv_scores.std() if len(cv_scores) > 0 else None,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': {
                'total': len(y),
                'inhibitors': int(y.sum()),
                'non_inhibitors': int(len(y) - y.sum())
            },
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"  训练准确率: {train_score:.3f}")
        print(f"  测试准确率: {test_score:.3f}")
        if auc_score:
            print(f"  ROC-AUC: {auc_score:.3f}")
        if cv_scores.mean():
            print(f"  交叉验证AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return results
    
    def predict(self, smiles):
        """
        预测单个分子
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            dict: 预测结果
        """
        if not self.model:
            return {'error': '模型未训练'}
        
        features = self.calculate_features(smiles)
        if not features:
            return {'error': '无效的SMILES'}
        
        # 准备特征向量
        X_desc = np.array([[features[col] for col in self.desc_cols]])
        X_fp = features['fingerprint'].reshape(1, -1)
        X = np.hstack([X_desc, X_fp])
        
        # 预测
        prob = self.model.predict_proba(X)[0, 1]
        pred = self.model.predict(X)[0]
        
        # 确定风险等级
        if prob >= 0.7:
            risk_level = 'High'
        elif prob >= 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'smiles': smiles,
            'cyp_isoform': self.cyp_isoform,
            'probability': float(prob),
            'prediction': int(pred),
            'risk_level': risk_level,
            'interpretation': self._get_interpretation(prob)
        }
    
    def batch_predict(self, smiles_list):
        """
        批量预测
        
        Args:
            smiles_list: SMILES列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for smiles in smiles_list:
            result = self.predict(smiles)
            results.append(result)
        return results
    
    def _get_interpretation(self, probability):
        """获取解释文本"""
        if probability >= 0.7:
            return f"高概率{self.cyp_isoform}抑制剂，可能引起药物相互作用"
        elif probability >= 0.3:
            return f"中等概率{self.cyp_isoform}抑制剂，需要注意"
        else:
            return f"低概率{self.cyp_isoform}抑制剂，风险较小"
    
    def save_model(self, model_dir='models'):
        """
        保存模型
        
        Args:
            model_dir: 模型保存目录
        """
        if not self.model:
            print("模型未训练，无法保存")
            return False
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{self.cyp_isoform.lower()}_model.pkl')
        
        model_data = {
            'model': self.model,
            'desc_cols': self.desc_cols,
            'cyp_isoform': self.cyp_isoform,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, model_path)
        print(f"模型已保存到: {model_path}")
        return True
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.desc_cols = model_data['desc_cols']
        self.cyp_isoform = model_data['cyp_isoform']
        self.feature_importance = model_data.get('feature_importance')
        
        print(f"模型已加载: {model_path}")
        return True


def train_all_cyp450_models(data_dir='data/cyp450/processed'):
    """
    训练所有CYP450模型
    
    Args:
        data_dir: 数据目录
        
    Returns:
        dict: 所有模型的结果
    """
    results = {}
    
    # CYP亚型列表
    cyp_isoforms = ['CYP3A4', 'CYP2D6', 'CYP2C9']
    
    for cyp in cyp_isoforms:
        print(f"\n{'='*60}")
        print(f"训练 {cyp} 模型")
        print(f"{'='*60}")
        
        # 查找数据文件
        data_files = [f for f in os.listdir(data_dir) 
                     if f.startswith(cyp.lower()) and f.endswith('.csv')]
        
        if not data_files:
            print(f"未找到 {cyp} 数据文件")
            continue
        
        # 使用最新的数据文件
        data_file = os.path.join(data_dir, sorted(data_files)[-1])
        print(f"使用数据文件: {data_file}")
        
        # 加载数据
        df = pd.read_csv(data_file)
        
        # 检查必要列
        required_cols = ['smiles', 'is_inhibitor']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"数据文件缺少列: {missing_cols}")
            continue
        
        # 训练模型
        predictor = CYP450Predictor(cyp_isoform=cyp)
        result = predictor.train(df)
        
        if 'error' not in result:
            # 保存模型
            predictor.save_model('models')
            results[cyp] = result
        else:
            print(f"训练失败: {result['error']}")
    
    return results


if __name__ == "__main__":
    print("CYP450预测模块测试")
    print("="*60)
    
    # 测试示例
    test_smiles = [
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # 布洛芬
        'CN1CCOC(=O)C(c2ccccc2)c2ccccc21',  # 酮康唑 (CYP3A4抑制剂)
        'COC1=CC=CC=C1OC',  # 帕罗西汀 (CYP2D6抑制剂)
    ]
    
    # 首先检查是否有数据并训练模型
    data_dir = 'data/cyp450/processed'
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print("发现CYP450数据，开始训练模型...")
        results = train_all_cyp450_models(data_dir)
        
        print("\n训练结果汇总:")
        for cyp, result in results.items():
            print(f"\n{cyp}:")
            print(f"  测试准确率: {result.get('test_accuracy', 'N/A'):.3f}")
            print(f"  AUC分数: {result.get('auc_score', 'N/A'):.3f}")
            print(f"  样本数: {result.get('n_samples', 'N/A')}")
    else:
        print("未找到CYP450数据，使用示例数据进行演示...")
        
        # 创建并训练一个演示模型
        from create_sample_data import create_sample_cyp450_data
        df = create_sample_cyp450_data()
        cyp3a4_df = df[df['cyp_isoform'] == 'CYP3A4']
        
        if len(cyp3a4_df) > 0:
            predictor = CYP450Predictor('CYP3A4')
            result = predictor.train(cyp3a4_df)
            
            print(f"\n演示模型训练结果:")
            print(f"  测试准确率: {result.get('test_accuracy', 'N/A'):.3f}")
            
            # 测试预测
            print(f"\n预测测试:")
            for smiles in test_smiles:
                prediction = predictor.predict(smiles)
                print(f"  {smiles[:30]}...: {prediction.get('risk_level', 'N/A')} "
                      f"(概率: {prediction.get('probability', 'N/A'):.3f})")
    
    print("\nCYP450预测模块测试完成!")
