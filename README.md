# GAMMA: Multi-Modal Glaucoma Classification

## 项目简介

GAMMA 是一个基于双分支多模态融合的青光眼自动分级深度学习框架。该模型同时利用**彩色眼底图像（CFP）**和**光学相干断层扫描图像（OCT）**进行青光眼的三分类诊断：

- **类别 0**: 无青光眼（Non-Glaucoma）
- **类别 1**: 早期青光眼（Early Glaucoma）
- **类别 2**: 中晚期青光眼（Mid-Advanced Glaucoma）

## 模型架构

### 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│                    DualBranchModel                          │
├─────────────────────────┬───────────────────────────────────┤
│     CFP 分支             │          OCT 分支                  │
│  ┌───────────────────┐  │  ┌─────────────────────────────┐   │
│  │  EfficientNet-B0  │  │  │   EfficientNet-B0           │   │
│  │  (特征提取)        │  │  │   (逐切片特征提取)          │   │
│  └─────────┬─────────┘  │  └──────────────┬──────────────┘   │
│            │            │                 │                 │
│            ▼            │                 ▼                 │
│  ┌───────────────────┐  │  ┌─────────────────────────────┐ │
│  │   CSPM             │  │  │   LA³ (Layer-prior         │ │
│  │   圆形结构感知模块  │  │  │    Augmented Axial         │ │
│  │   (多半径软圆形核)  │  │  │    Attention)               │ │
│  └─────────┬─────────┘  │  └──────────────┬──────────────┘   │
│            │            │                 │                 │
│            │            │                 ▼                 │
│            │            │         [B×K, 1280, H', W']      │
│            │            │                 │                 │
│            │            │                 ▼                 │
│            │            │      AdaptiveAvgPool(K维均值)     │
│            │            │                 │                 │
│            │            └────────────[B, 1280]─────────────┤
│            │                              │                │
│            └──────────────┐    ┌───────────┘                │
│                           ▼    ▼                            │
│                   ┌─────────────────┐                       │
│                   │   CSSF          │                       │
│                   │   条件状态空间融合 │                       │
│                   │   (FiLM条件化)   │                       │
│                   └────────┬────────┘                       │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │  分类头 (3类)    │                       │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块

#### 1. CSPM (Circular-Structure Perception Module)
**圆形结构感知模块**，用于增强眼底图像的空间注意力感知能力：
- **通道注意力**: 基于全局平均池化和最大池化的通道权重学习
- **空间注意力**: 采用**多半径软圆形核**设计，三个并行分支分别学习不同半径的圆形结构模式
- 更好地捕捉眼底图像中的视杯、视盘等圆形结构特征

#### 2. LA³ (Layer-prior Augmented Axial Attention)
**层先验增强轴向注意力**，专为OCT图像设计：
- **H轴分支**: 带层先验偏置的轴向注意力，利用k_h×1条带卷积从原始特征图计算层结构感知权重
- **W轴分支**: 标准2019年轴向注意力机制
- 有效建模OCT图像中视网膜各层的结构特征

#### 3. CSSF (Conditional State-Space Fusion)
**条件状态空间融合模块**，实现CFP与OCT特征的深度融合：
- **FiLM条件化**: 由OCT全局向量生成调制参数（γ, β）
- **选择性扫描**: 基于2D状态空间模型的双向（行+列）选择性扫描
- **动态参数**: 条件化的时间步长Δdt和状态矩阵A参数

## 数据集

使用GAMMA竞赛数据集，包含：
- **训练集**: `dataset/Train/`
- **验证集**: `dataset/Validation/`
- **测试集**: `dataset/Test/`

每位患者同时具有：
- 1张彩色眼底图像（CFP）
- 多张OCT切片图像

## 训练配置

| 参数 | 值 |
|------|-----|
| 批量大小 (Batch Size) | 4 |
| 训练轮数 (Epochs) | 50 |
| 初始学习率 | 0.0001 |
| 优化器 | Adam |
| 学习率调度 | ExponentialLR (γ=0.95) |
| 损失函数 | Focal Loss (γ=2, 加权) |
| 图像尺寸 | 512×512 |
| OCT切片数量 | 16张 |

### 数据增强

**眼底图像 (CFP)**:
- 随机水平翻转
- 随机旋转 (±30°)
- 中心裁剪 (512)

**OCT图像**:
- 随机水平翻转
- 随机仿射变换 (平移±5%, 缩放0.94-1.06)

## 评估指标

模型训练完成后自动输出以下指标：

- **Accuracy**: 分类准确率
- **Kappa**: Cohen's Kappa系数（加权）
- **Macro-F1**: 宏观平均F1分数
- **Weighted-F1**: 加权F1分数
- **Macro-Recall/Precision/Specificity**: 宏观平均召回率/精确率/特异性
- **Macro-AUC**: 宏观平均ROC-AUC

## 可视化分析

训练完成后自动生成：

1. **ROC曲线** (`roc_curve.png`): 各类别的ROC曲线及宏观平均AUC
2. **混淆矩阵** (`confusion_matrix.png`): 分类结果的热力图展示
3. **T-SNE可视化** (`tsne_fusion_visualization.png`): 融合特征的二维可视化
4. **GradCAM热力图** (可选): CFP分支、OCT分支及融合模块的注意力可视化

## 快速开始

### 环境依赖

```bash
pip install torch torchvision
pip install pytorch-grad-cam
pip install einops
pip install scikit-learn matplotlib pandas
```

### 运行训练

```bash
# 设置GPU
export CUDA_VISIBLE_DEVICES=6

# 直接运行
python MCLC_k=5.py

# 或指定日志目录
export LOG_DIR=/path/to/log_directory
python MCLC_k=5.py
```

## 项目结构

```
GAMMA/code/
├── MCLC_k=5.py              # 主训练脚本
├── MODEL/
│   ├── Attention.py          # CSPM模块定义
│   ├── AX.py                 # LA³模块定义
│   ├── Fusion.py             # CSSF融合模块定义
│   ├── dataloaderMulti.py    # 双模态数据加载器
│   └── loss.py               # Focal Loss定义
└── result/                   # 训练日志和可视化结果
    └── MCLC_k_5/
        ├── training_log.txt
        ├── best_model_*.pth
        ├── roc_curve.png
        ├── confusion_matrix.png
        └── tsne_fusion_visualization.png
```

## 技术亮点

1. **多模态融合**: 首次将CFP的圆形结构感知与OCT的层结构感知相结合
2. **状态空间建模**: CSSF模块利用选择性扫描机制实现高效的长程依赖建模
3. **条件化设计**: FiLM条件化使OCT特征能够动态调制CFP特征的处理过程
4. **可解释性强**: 支持GradCAM可视化，便于理解模型决策依据

## 引用

如果你在研究中使用了本代码或其中部分模块，请引用相关工作。
