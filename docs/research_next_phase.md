# PharmaAI 下一阶段开发方向调研报告

> 文档生成时间: 2026-03-14  
> 调研范围: 深度学习、模型服务化、批处理优化、功能扩展、行业标准工具  
> 目标: 为 PharmaAI 项目 3 个月技术演进提供决策依据

---

## 一、技术趋势总结

### 1.1 深度学习方向 - 图神经网络 (GNN)

#### 技术现状

**GNN 在药物发现中的应用已趋于成熟**，是当前分子性质预测的主流深度学习方案：

| 架构 | 特点 | 代表工作 |
|------|------|----------|
| **GIN** (Graph Isomorphism Network) | 区分图同构能力强，理论可证明 | Xu et al., ICLR 2019 |
| **GAT** (Graph Attention Network) | 引入注意力机制，可解释性强 | Veličković et al., ICLR 2018 |
| **MPNN** (Message Passing NN) | 化学领域应用最广泛 | Gilmer et al., ICML 2017 |
| **Transformer-based** | 捕获长程依赖，参数量大 | Graphormer, Transformer-M |

**SOTA 效果对比**（MoleculeNet 基准）：
- Chemprop (MPNN) 在多个任务上表现优异
- 3D Infomax 预训练可将 MAE 降低 22%（量子力学性质）
- 自监督预训练（如 GraphCL、InfoGraph）显著提升小样本场景性能

**与 RDKit 集成方案**：
- PyTorch Geometric 提供 `torch_geometric.utils.from_smiles()` 直接转换
- DeepChem 内置 `ConvMol` 和 `Weave` 转换器
- DGL-LifeSci 提供 RDKit 友好的数据加载器

#### 实际应用案例
1. **Atomwise** - 使用 3D CNN + GNN 进行虚拟筛选
2. **Insilico Medicine** - 基于 GAN + GNN 的分子生成
3. **Exscientia** - 图神经网络用于 ADMET 预测

---

### 1.2 模型服务化 - FastAPI/ML Serving

#### 技术现状

**FastAPI 已成为 Python ML 服务化的事实标准**：

| 框架 | 并发性能 | 易用性 | 社区活跃度 | 适用场景 |
|------|----------|--------|------------|----------|
| **FastAPI** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 生产环境首选 |
| **Flask** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 原型开发 |
| **Django** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 全栈应用 |
| **BentoML** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 模型部署专用 |
| **Ray Serve** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 大规模分布式 |

**性能对比**（基于 TechEmpower 基准）：
- FastAPI + Uvicorn: ~18,000 req/s
- Flask + Gunicorn: ~2,000 req/s
- Django: ~1,200 req/s

**MLflow 集成趋势**：
- MLflow 2.x 原生支持模型签名和输入验证
- `mlflow.pyfunc` 可直接包装 sklearn 模型
- 与 FastAPI 通过 `mlflow.pyfunc.load_model()` 无缝集成
- 支持模型版本控制和 A/B 测试

#### 生产部署最佳实践
1. **容器化**: Docker + FastAPI + Uvicorn (workers > 1)
2. **模型加载**: 启动时预加载，避免请求时冷启动
3. **批处理 API**: 支持批量预测减少开销
4. **监控**: 集成 Prometheus + Grafana
5. **限流**: 使用 slowapi 或 nginx 限流

---

### 1.3 批处理优化 - Dask/Ray

#### 技术现状

**Dask 和 Ray 的差异化定位**：

| 特性 | Dask | Ray |
|------|------|-----|
| **设计理念** | 扩展 pandas/numpy API | 通用分布式计算框架 |
| **学习曲线** | 低（pandas 用户友好） | 中等 |
| **ML 集成** | 优秀 (dask-ml, xgboost) | 优秀 (Ray Train, Tune) |
| **任务调度** | 静态图调度 | 动态任务调度 |
| **容错能力** | 中等 | 强 |
| **内存管理** | 显式分区控制 | 对象存储 (Plasma) |

**RDKit 并行处理方案**：
- **multiprocessing**: 简单但进程开销大
- **Dask Bag**: 适合不规则数据（SMILES 字符串）
- **Ray**: 适合复杂依赖的任务图
- **Joblib**: sklearn 原生支持，可无缝切换到 Ray backend

**性能基准**（10万分子指纹生成）：
- 单线程: ~300s
- multiprocessing (8核): ~45s
- Dask (8核): ~40s
- Ray (8核): ~35s

#### 内存优化策略
1. **分块处理**: 使用 Dask 的 `map_partitions`
2. **惰性求值**: 延迟计算直到 `compute()` 调用
3. **数据类型优化**: 使用 `float32` 替代 `float64`
4. **特征缓存**: 缓存常见分子的指纹结果

---

### 1.4 功能扩展方向

#### 技术现状

**分子生成模型（Molecular Generative Models）**：

| 模型类型 | 代表方法 | 优势 | 局限 |
|----------|----------|------|------|
| **VAE** | Junction Tree VAE, Grammar VAE | 连续潜在空间，可插值 | 生成分子质量一般 |
| **GAN** | ORGAN, MolGAN | 生成多样性高 | 训练不稳定 |
| **Flow-based** | MoFlow, GraphAF | 精确似然计算 | 模型复杂度高 |
| **RL-based** | REINVENT, GCPN | 可优化特定性质 | 需要奖励函数设计 |
| **Diffusion** | DiGress, GeoLDM (2024) | SOTA 效果 | 计算成本高 |

**多任务学习（Multi-task Learning）**：
- 共享表示学习多个 ADMET 性质
- 硬参数共享 vs 软参数共享（如 MMOE）
- 在 ChEMBL 数据集上 MTL 比单任务提升 10-15%

**药物-靶点相互作用（DTI）预测**：
- DeepConv-DTI: CNN + 序列编码
- GraphDTA: GNN + Transformer
- MONN: 分子-蛋白质联合表示

**强化学习在分子优化中的应用**：
- REINVENT 4.0（2024）支持多目标优化
- 结合化学约束（SA score, QED）的奖励设计
- 与分子对接工具集成进行虚拟筛选

---

### 1.5 行业标准与工具

#### 技术现状

**化学数据库 API**：

| 数据库 | API 类型 | 数据规模 | 访问方式 |
|--------|----------|----------|----------|
| **ChEMBL** | REST API | >200万化合物 | 需要 API Key |
| **PubChem** | PUG-REST | >1亿化合物 | 公开访问，限流 |
| **DrugBank** | REST API | ~15,000药物 | 需要订阅 |
| **ZINC** | 批量下载 | >10亿化合物 | 免费 |

**分子对接工具**：

| 工具 | 类型 | 性能 | Python 接口 |
|------|------|------|-------------|
| **Gnina** | 开源深度学习对接 | 接近 AutoDock | pygnina |
| **AutoDock Vina** | 开源传统对接 | 行业标准 | meeko, vina |
| **SMINA** | AutoDock 优化版 | 比 Vina 快 2x | 命令行 |
| **DiffDock** | 深度学习对接 (2023) | SOTA 效果 | 开源 |

**行业标准数据格式**：
- **SMILES**: 线性表示，最常用
- **SDF/MOL**: 2D/3D 结构，包含原子坐标
- **InChI**: 标准化标识符
- **SMI**: 简化 SMILES 格式

---

## 二、推荐技术栈

### 2.1 深度学习模块

```
推荐方案: PyTorch Geometric + RDKit

核心依赖:
├── torch>=2.0.0
├── torch-geometric>=2.4.0
├── torch-scatter, torch-sparse (性能优化)
├── rdkit>=2023.09
└── ogb (Open Graph Benchmark)

可选增强:
├── deepchem>=2.7.0 (MoleculeNet 数据集)
├── dgl>=1.1.0 (替代 PyG)
└── transformers (用于序列模型)
```

**推荐 GNN 架构组合**：
1. **基础模型**: GIN (Graph Isomorphism Network) - 稳健、高效
2. **注意力增强**: GATv2 - 动态注意力，更好的表达能力
3. **预训练**: 3D Infomax 或 GraphCL 自监督预训练

### 2.2 模型服务化架构

```
推荐方案: FastAPI + MLflow

核心组件:
├── FastAPI (API 框架)
├── Uvicorn (ASGI 服务器)
├── MLflow (模型注册与管理)
├── Pydantic v2 (数据验证)
└── Prometheus + Grafana (监控)

部署选项:
├── 单机: Docker Compose
├── 集群: Kubernetes + Helm
└── 云原生: AWS SageMaker / GCP Vertex AI
```

**服务架构建议**：
```
Client Request → FastAPI → MLflow Model Registry → Predictor
                                    ↓
                              Model Version Control
                                    ↓
                              Prometheus Metrics
```

### 2.3 批处理优化方案

```
推荐方案: Dask (本地) / Ray (分布式)

选择策略:
├── 单节点 (< 100万分子): Dask + Joblib
├── 多节点/云环境: Ray + Ray Train
└── 超大规模: Ray + Dask-on-Ray

集成方式:
├── dask-ml (与 sklearn 兼容)
├── ray.util.joblib (替换 joblib backend)
└── dask.distributed (本地集群)
```

### 2.4 功能扩展技术选型

```
分子生成:
├── REINVENT 4.0 (首选，开源，维护活跃)
├── MolGPT (Transformer 基础)
└── 自研 VAE (基于 PyTorch Geometric)

DTI 预测:
├── DeepPurpose (统一框架)
├── DeepConv-DTI (轻量级)
└── GraphDTA (GNN-based)

分子对接:
├── PyAutoDock (Python 包装)
├── Smina (命令行集成)
└── Gnina (深度学习对接)
```

### 2.5 数据与工具链

```
数据来源:
├── ChEMBL (生物活性数据)
├── PubChem (化学结构)
├── DrugBank (药物信息)
└── MoleculeNet (基准数据集)

工具集成:
├── RDKit (化学信息学核心)
├── Open Babel (格式转换)
├── OpenMM (分子动力学)
└── PDBFixer (蛋白质结构处理)
```

---

## 三、实现建议

### 3.1 GNN 模块集成方案

**阶段一：基础 GNN 预测器**（4周）

```python
# pharmaai/core/gnn_predictor.py
from abc import ABC
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
from pharmaai.core.base_predictor import BasePredictor

class GNNPredictor(BasePredictor):
    """基于 GNN 的分子性质预测器"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _smiles_to_graph(self, smiles: str) -> Data:
        """将 SMILES 转换为 PyG Data 对象"""
        from torch_geometric.utils import from_smiles
        return from_smiles(smiles)
    
    def predict(self, smiles: str) -> PredictionResult:
        data = self._smiles_to_graph(smiles).to(self.device)
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index, data.batch)
        return self._format_result(pred)
```

**阶段二：预训练模型支持**（2周）
- 集成 3D Infomax 预训练权重
- 支持迁移学习

### 3.2 FastAPI 服务集成

**项目结构**：
```
pharmaai/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI 应用入口
│   ├── routes/
│   │   ├── predict.py    # 预测 API
│   │   ├── batch.py      # 批处理 API
│   │   └── health.py     # 健康检查
│   ├── models/
│   │   └── schemas.py    # Pydantic 模型
│   └── dependencies.py   # 依赖注入
└── serving/
    ├── model_manager.py  # 模型生命周期管理
    └── mlflow_client.py  # MLflow 集成
```

**核心实现**：
```python
# pharmaai/api/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pharmaai.serving.model_manager import ModelManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载所有模型
    app.state.model_manager = ModelManager()
    await app.state.model_manager.load_all()
    yield
    # 关闭时清理资源
    await app.state.model_manager.cleanup()

app = FastAPI(
    title="PharmaAI API",
    version="0.3.0",
    lifespan=lifespan
)

# pharmaai/api/routes/predict.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/predict", tags=["prediction"])

class PredictRequest(BaseModel):
    smiles: str
    model_name: str = "default"
    model_version: str = "latest"

class PredictResponse(BaseModel):
    predictions: dict
    model_version: str
    inference_time_ms: float

@router.post("/single", response_model=PredictResponse)
async def predict_single(
    request: PredictRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    predictor = model_manager.get_predictor(
        request.model_name, 
        request.model_version
    )
    result = predictor.predict(request.smiles)
    return PredictResponse(
        predictions=result.to_dict(),
        model_version=request.model_version,
        inference_time_ms=result.elapsed_ms
    )
```

### 3.3 MLflow 集成方案

```python
# pharmaai/serving/mlflow_client.py
import mlflow
from mlflow.tracking import MlflowClient

class MLflowModelRegistry:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(self, model, name: str, metrics: dict = None):
        """注册模型到 MLflow"""
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                name
            )
    
    def load_model(self, name: str, version: str = "latest"):
        """从 MLflow 加载模型"""
        model_uri = f"models:/{name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
```

### 3.4 Dask/Ray 批处理集成

**Dask 方案**（推荐用于单节点）：
```python
# pharmaai/core/batch_processor.py
import dask.bag as db
from dask.distributed import Client
from typing import List, Callable

class DaskBatchProcessor:
    def __init__(self, n_workers: int = None):
        self.client = Client(n_workers=n_workers)
    
    def process_smiles_batch(
        self, 
        smiles_list: List[str], 
        processor: Callable,
        batch_size: int = 1000
    ) -> List[dict]:
        """并行处理 SMILES 列表"""
        bag = db.from_sequence(smiles_list, partition_size=batch_size)
        results = bag.map(processor).compute()
        return results
    
    def generate_fingerprints_parallel(
        self, 
        smiles_list: List[str]
    ) -> np.ndarray:
        """并行生成分子指纹"""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        def _fp(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        
        bag = db.from_sequence(smiles_list)
        fingerprints = bag.map(_fp).compute()
        return np.array([fp for fp in fingerprints if fp is not None])
```

**Ray 方案**（推荐用于分布式）：
```python
# pharmaai/core/ray_processor.py
import ray
from ray.util.joblib import register_ray
from sklearn.model_selection import cross_val_score

class RayBatchProcessor:
    def __init__(self, address: str = None):
        if not ray.is_initialized():
            ray.init(address=address)
        register_ray()
    
    @ray.remote
    def _predict_batch(self, model, smiles_batch):
        """远程预测任务"""
        return [model.predict(s) for s in smiles_batch]
    
    def distributed_predict(
        self, 
        model, 
        smiles_list: List[str], 
        n_workers: int = 4
    ) -> List:
        """分布式批量预测"""
        batch_size = len(smiles_list) // n_workers
        batches = [
            smiles_list[i:i + batch_size] 
            for i in range(0, len(smiles_list), batch_size)
        ]
        
        model_ref = ray.put(model)  # 共享模型到对象存储
        futures = [
            self._predict_batch.remote(model_ref, batch) 
            for batch in batches
        ]
        results = ray.get(futures)
        return [r for batch in results for r in batch]
```

### 3.5 分子生成功能集成

```python
# pharmaai/core/molecule_generator.py
from abc import ABC, abstractmethod
from typing import List

class MoleculeGenerator(ABC):
    """分子生成器抽象基类"""
    
    @abstractmethod
    def generate(
        self, 
        n_samples: int, 
        constraints: dict = None
    ) -> List[str]:
        """生成分子 SMILES 列表"""
        pass
    
    @abstractmethod
    def optimize(
        self, 
        seed_smiles: str, 
        target_property: str,
        n_steps: int = 100
    ) -> List[str]:
        """基于性质的分子优化"""
        pass

class REINVENTGenerator(MoleculeGenerator):
    """基于 REINVENT 的分子生成器"""
    
    def __init__(self, model_path: str, config: dict):
        from reinvent_models.libinvent.models import model as reinvent_model
        self.model = reinvent_model.load_from_file(model_path)
        self.config = config
    
    def generate(self, n_samples: int, constraints: dict = None) -> List[str]:
        # 调用 REINVENT API 生成分子
        pass
    
    def optimize(
        self, 
        seed_smiles: str, 
        target_property: str,
        n_steps: int = 100
    ) -> List[str]:
        # 使用 RL 优化分子
        pass
```

---

## 四、优先级建议

### 4.1 优先级矩阵

```
                    业务价值
         低          中          高
       ┌─────────┬─────────┬─────────┐
    高 │  Ray    │  MLflow │  FastAPI│
       │  分布式 │  集成   │  服务化 │
 技    ├─────────┼─────────┼─────────┤
 术    │  GNN    │  Dask   │  GNN    │
 成    │  预训练 │  批处理 │  基础   │
 熟    ├─────────┼─────────┼─────────┤
 度    │  DTI    │  分子   │  API    │
    低 │  预测   │  生成   │  文档   │
       └─────────┴─────────┴─────────┘
```

### 4.2 详细优先级规划

#### P0 - 立即启动（第1个月）

| 任务 | 价值 | 工作量 | 交付物 |
|------|------|--------|--------|
| FastAPI 服务开发 | ⭐⭐⭐⭐⭐ | 3周 | REST API + 文档 |
| MLflow 模型管理 | ⭐⭐⭐⭐ | 2周 | 模型注册/版本控制 |
| Dask 批处理 | ⭐⭐⭐⭐ | 2周 | 并行指纹生成 |

#### P1 - 短期跟进（第2个月）

| 任务 | 价值 | 工作量 | 交付物 |
|------|------|--------|--------|
| GNN 基础模型 | ⭐⭐⭐⭐⭐ | 4周 | GIN 毒性预测器 |
| 性能基准测试 | ⭐⭐⭐ | 1周 | 性能报告 |
| CI/CD 完善 | ⭐⭐⭐ | 1周 | GitHub Actions |

#### P2 - 中期规划（第3个月）

| 任务 | 价值 | 工作量 | 交付物 |
|------|------|--------|--------|
| GNN 预训练 | ⭐⭐⭐⭐ | 3周 | 自监督预训练模型 |
| 分子生成 MVP | ⭐⭐⭐ | 2周 | REINVENT 集成 |
| Ray 分布式支持 | ⭐⭐⭐ | 2周 | 多节点批处理 |

#### P3 - 长期考虑（3个月以后）

- DTI 预测功能
- 分子对接集成
- 3D 构象生成
- 可视化界面增强

### 4.3 技术债务与风险考量

**需要关注的风险**：
1. **GNN 模型可解释性**: 提供 SHAP 或注意力可视化
2. **模型版本兼容性**: 使用 MLflow 的模型签名功能
3. **服务高可用**: FastAPI 支持优雅关闭和健康检查
4. **数据隐私**: 批处理时注意敏感数据脱敏

---

## 五、参考资源

### 5.1 重要论文

**GNN 基础与药物发现**：
1. Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. ICML.
2. Xu, K., et al. (2019). How powerful are graph neural networks? ICLR.
3. Stärk, H., et al. (2022). 3D Infomax improves GNNs for Molecular Property Prediction. ICML.
4. Yang, K., et al. (2019). Analyzing learned molecular representations for property prediction. JCIM.

**分子生成**：
5. Bjerrum, E., & Threlfall, R. (2017). Molecular generation with recurrent neural networks. arXiv.
6. Blaschke, T., et al. (2020). REINVENT 2.0: an AI tool for de novo drug design. JCIM.
7. Gao, W., & Coley, C. W. (2020). The synthesizability of molecules proposed by generative models. JCIM.

**药物-靶点相互作用**：
8. Öztürk, H., et al. (2018). DeepDTA: deep drug-target binding affinity prediction. Bioinformatics.
9. Nguyen, T., et al. (2021). GraphDTA: prediction of drug-target binding affinity using graph convolutional networks. Bioinformatics.

### 5.2 GitHub 项目

**核心框架**：
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - 图神经网络库
- [DeepChem](https://github.com/deepchem/deepchem) - 化学深度学习
- [RDKit](https://github.com/rdkit/rdkit) - 化学信息学工具包
- [Chemprop](https://github.com/chemprop/chemprop) - 消息传递神经网络

**模型服务化**：
- [FastAPI](https://github.com/tiangolo/fastapi) - 高性能 Web 框架
- [MLflow](https://github.com/mlflow/mlflow) - ML 生命周期管理
- [BentoML](https://github.com/bentoml/bentoml) - 模型服务化平台

**批处理与分布式**：
- [Dask](https://github.com/dask/dask) - 并行计算库
- [Ray](https://github.com/ray-project/ray) - 分布式计算框架

**分子生成**：
- [REINVENT 4.0](https://github.com/MolecularAI/Reinvent) - 强化学习分子生成
- [MolGPT](https://github.com/devalab/molgpt) - 分子生成 GPT
- [JTVAE](https://github.com/wengong-jin/icml18-jtnn) - 连接树 VAE

**数据集与基准**：
- [MoleculeNet](https://moleculenet.org/) - 分子机器学习基准
- [Open Graph Benchmark](https://ogb.stanford.edu/) - 图学习基准

### 5.3 教程与文档

**官方文档**：
- [PyTorch Geometric 教程](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
- [DeepChem 教程](https://deepchem.readthedocs.io/en/latest/get_started/tutorials.html)
- [FastAPI 教程](https://fastapi.tiangolo.com/tutorial/)
- [MLflow 文档](https://mlflow.org/docs/latest/index.html)

**在线课程**：
- Stanford CS224W: Machine Learning with Graphs
- DeepLearning.AI: AI for Drug Discovery

**博客文章**：
- [Chemprop 论文解读](https://chemprop.readthedocs.io/)
- [GNN 在药物发现中的应用](https://towardsdatascience.com/graph-neural-networks-for-drug-discovery-4db33e327ada)

---

## 六、总结与行动计划

### 6.1 核心建议

1. **优先构建 FastAPI + MLflow 服务架构** - 这是提供生产就绪 API 的基础
2. **并行开展 GNN 基础模型研发** - 利用 PyTorch Geometric 快速验证
3. **使用 Dask 解决当前批处理瓶颈** - 立即可实施的性能优化
4. **预留分子生成和 DTI 的扩展接口** - 为未来功能扩展做准备

### 6.2 3 个月里程碑

```
Month 1 (v0.3.0-alpha):
├── FastAPI REST API 上线
├── MLflow 模型注册完成
├── Dask 批处理优化完成
└── GNN 基础模型原型

Month 2 (v0.3.0-beta):
├── GNN 毒性预测器上线
├── API 性能优化 (目标: <10ms/分子)
├── 集成测试覆盖 80%+
└── 性能基准报告

Month 3 (v0.3.0):
├── GNN 预训练模型集成
├── 分子生成 MVP 发布
├── 完整文档和教程
└── 生产部署就绪
```

### 6.3 下一步行动

1. **本周**: 创建 FastAPI 项目脚手架，定义 API 规范
2. **下周**: 集成 MLflow，完成模型注册 POC
3. **第3周**: 并行开发 GNN 基础模型和 Dask 批处理
4. **第4周**: 集成测试，性能调优

---

**文档版本**: 1.0  
**最后更新**: 2026-03-14  
**下次审查**: 与架构师讨论后更新
