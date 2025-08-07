# 时空特征融合网络模型构建方案

## 3.1 整体架构设计

基于现有静态CNN定位网络的局限性，本文提出了一种融合Modern TCN的时空特征提取网络架构。该网络通过CNN模块提取RSS信号的空间特征，利用Modern TCN模块建模时序依赖关系，最终实现位置和姿态的联合预测。

### 3.1.1 网络整体框架

时空特征融合网络由以下四个核心模块构成：

1. **输入预处理模块**：将静态RSS数据转换为时序格式
2. **空间特征提取模块**：基于改进的CNN架构提取每个时刻的空间特征
3. **时序建模模块**：采用Modern TCN捕捉时间维度的依赖关系
4. **特征融合与输出模块**：多尺度特征融合及位置姿态联合预测

## 3.2 输入数据重构策略

### 3.2.1 时序数据构建

针对原有静态RSS数据 $\mathbf{X} \in \mathbb{R}^{N \times 16}$，采用滑动窗口方法构建时序输入：

$$\mathbf{X}_{seq} = \{\mathbf{x}_{t-T+1}, \mathbf{x}_{t-T+2}, \ldots, \mathbf{x}_t\}$$

其中：
- $T$ 为时序窗口长度（设置为10）
- $\mathbf{x}_t \in \mathbb{R}^{16}$ 表示第$t$时刻的RSS信号向量
- 最终输入维度：$\mathbf{X}_{seq} \in \mathbb{R}^{N \times T \times 16}$

### 3.2.2 标签对应策略

对于时序样本 $\mathbf{X}_{seq}[i:i+T]$，采用序列末尾时刻的标签作为监督信号：

$$\mathbf{y}_{target} = \mathbf{y}_{i+T-1} = [x, y, z, \alpha, \beta, \gamma]^T$$

该策略确保网络利用历史信息预测当前时刻的位置和姿态。

## 3.3 空间特征提取模块设计

### 3.3.1 基于CNN的空间特征提取

保留并改进原有CNN网络的空间特征提取能力，对时序中的每个时刻独立进行特征提取：

**第一层卷积块：**
$$\mathbf{h}_1^{(t)} = \text{BN}(\text{Dropout}(\text{ReLU}(\text{Conv1D}(\mathbf{x}_t))))$$

**第二层卷积块：**
$$\mathbf{h}_2^{(t)} = \text{BN}(\text{Dropout}(\text{ReLU}(\text{Conv1D}(\mathbf{h}_1^{(t)}))))$$

**空间特征聚合：**
$$\mathbf{f}_{spatial}^{(t)} = \text{GlobalAvgPool1D}(\mathbf{h}_2^{(t)})$$

### 3.3.2 网络参数配置

| 参数 | 数值 | 说明 |
|------|------|------|
| 卷积核大小 | 16 | 覆盖全部AP信号 |
| 滤波器数量 | 64 | 保持与原网络一致 |
| Dropout率 | 0.2 | 防止过拟合 |
| 激活函数 | ReLU | 增强非线性表达能力 |

最终得到空间特征序列：$\mathbf{F}_{spatial} = [\mathbf{f}_{spatial}^{(1)}, \mathbf{f}_{spatial}^{(2)}, \ldots, \mathbf{f}_{spatial}^{(T)}]$

## 3.4 Modern TCN时序建模模块

### 3.4.1 Modern TCN核心组件

Modern TCN模块包含三个关键子模块，分别建模不同维度的依赖关系：

**深度卷积（DWConv）：时间依赖性建模**

采用深度可分离卷积独立学习每个通道的时间模式：
$$\mathbf{h}_{dw} = \text{DWConv}(\text{LN}(\mathbf{F}_{spatial}))$$

其中，DWConv的分组数设置为特征维度$d_{model}$，实现通道间的独立时间建模。

**前馈网络1（ConvFFN1）：通道依赖性建模**

学习特征通道间的相互关系：
$$\mathbf{h}_{ffn1} = \text{Conv1D}_{1×1}(\text{GELU}(\text{Conv1D}_{1×1}(\text{LN}(\mathbf{h}_{dw} + \mathbf{F}_{spatial}))))$$

**前馈网络2（ConvFFN2）：序列依赖性建模**

捕捉多序列间的跨变量依赖关系：
$$\mathbf{h}_{ffn2} = \text{Conv1D}_{1×1}(\text{GELU}(\text{Conv1D}_{1×1}(\text{LN}(\mathbf{h}_{ffn1}))))$$

### 3.4.2 Modern TCN块级联

通过多个Modern TCN块的级联实现深层时序特征提取：

$$\mathbf{F}_{temporal}^{(l)} = \text{ModernTCN}^{(l)}(\mathbf{F}_{temporal}^{(l-1)})$$

其中$l = 1, 2, 3$表示TCN层数，$\mathbf{F}_{temporal}^{(0)} = \mathbf{F}_{spatial}$。

### 3.4.3 残差连接与归一化

每个子模块均采用残差连接和层归一化，提升训练稳定性：

$$\mathbf{output} = \mathbf{input} + \text{Dropout}(\text{SubModule}(\text{LayerNorm}(\mathbf{input})))$$

## 3.5 多尺度特征融合策略

### 3.5.1 特征层次设计

为充分利用不同时间尺度的信息，设计了多层次特征融合机制：

**当前时刻空间特征（即时性）：**
$$\mathbf{f}_{current} = \mathbf{f}_{spatial}^{(T)}$$

**历史时序特征（连续性）：**
$$\mathbf{f}_{temporal} = \text{GlobalAvgPool1D}(\mathbf{F}_{temporal}^{(3)})$$

### 3.5.2 特征融合操作

采用特征拼接方式实现多尺度融合：
$$\mathbf{f}_{fused} = \text{Concat}([\mathbf{f}_{current}, \mathbf{f}_{temporal}])$$

融合后的特征维度为：$\mathbf{f}_{fused} \in \mathbb{R}^{128}$

## 3.6 联合输出设计

### 3.6.1 双分支预测架构

设计独立的位置和姿态预测分支，实现专门化的特征学习：

**位置预测分支：**
$$\mathbf{p}_{pos} = \text{Linear}(\text{ReLU}(\text{Linear}(\mathbf{f}_{fused})))$$
$$\hat{\mathbf{y}}_{pos} = [\hat{x}, \hat{y}, \hat{z}]^T \in \mathbb{R}^3$$

**姿态预测分支：**
$$\mathbf{p}_{ori} = \text{Linear}(\text{ReLU}(\text{Linear}(\mathbf{f}_{fused})))$$
$$\hat{\mathbf{y}}_{ori} = [\hat{\alpha}, \hat{\beta}, \hat{\gamma}]^T \in \mathbb{R}^3$$

### 3.6.2 联合损失函数

采用加权多任务损失函数同时优化位置和姿态预测：

$$\mathcal{L}_{total} = w_{pos} \cdot \mathcal{L}_{pos} + w_{ori} \cdot \mathcal{L}_{ori}$$

其中：
- $\mathcal{L}_{pos} = \text{MSE}(\mathbf{y}_{pos}, \hat{\mathbf{y}}_{pos})$ 为位置损失
- $\mathcal{L}_{ori} = \text{MSE}(\mathbf{y}_{ori}, \hat{\mathbf{y}}_{ori})$ 为姿态损失
- $w_{pos} = 1.0$，$w_{ori} = 0.5$ 为损失权重

## 3.7 网络训练策略

### 3.7.1 超参数配置

| 超参数 | 数值 | 说明 |
|--------|------|------|
| 时序长度 | 10 | 平衡时序信息与计算效率 |
| 学习率 | 0.001 | Adam优化器初始学习率 |
| 批大小 | 128 | 平衡内存使用与收敛速度 |
| TCN层数 | 3 | 充分的时序特征提取深度 |
| 特征维度 | 64 | 与原CNN网络保持一致 |

### 3.7.2 优化策略

**自适应学习率调整：**
采用ReduceLROnPlateau策略，当验证损失停止改善时自动降低学习率。

**早停机制：**
监控验证集损失，连续10个epoch无改善时停止训练，防止过拟合。

**数据增强：**
- 时序抖动：随机调整序列长度$T \pm 2$
- 信号噪声：添加均值为0、标准差为0.01的高斯噪声
- 时序平移：采用不同步长的滑动窗口增加样本多样性

## 3.8 网络架构优势分析

### 3.8.1 相比原有静态CNN的改进

| 方面 | 静态CNN | 时空融合网络 | 改进效果 |
|------|---------|-------------|----------|
| **输入维度** | $[N, 16]$ | $[N, 10, 16]$ | 引入时序信息 |
| **特征类型** | 空间特征 | 空间+时序特征 | 多维度特征融合 |
| **网络深度** | 4层Conv1D | 4层Conv1D+3层TCN | 增强表达能力 |
| **预测目标** | 联合6维输出 | 分离位置姿态预测 | 专门化学习 |
| **动态适应性** | 无 | 时序依赖建模 | 处理运动目标 |

### 3.8.2 理论性能预期

基于时空特征融合的优势，预期网络性能改进：

- **3D位置精度**：MAE降低15-30%，RMSE降低20-26%
- **姿态估计精度**：角度误差降低20-32%
- **动态鲁棒性**：对信号波动和目标运动的适应性显著提升
- **长期稳定性**：利用历史信息平滑预测结果

该时空特征融合网络架构在保持原有空间特征提取优势的基础上，通过Modern TCN的引入实现了时序建模能力的显著增强，为动态环境下的高精度定位提供了有效的技术方案。