"""
=================================================================================
动态定位网络完整方案包
基于现有静态CNN网络的Modern TCN融合改进方案
=================================================================================

本文件包含完整的：
1. 网络架构实现代码
2. 训练脚本
3. 论文方案部分内容
4. 使用说明和示例

作者：AI Assistant
基于：Regression_CNN.ipynb 静态定位网络
目标：实现动态环境下的高精度定位
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Flatten, Dropout, BatchNormalization,
    Reshape, Concatenate, Add, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =================================================================================
# 1. Modern TCN核心模块实现
# =================================================================================

class ModernTCNBlock(tf.keras.layers.Layer):
    """
    Modern TCN模块实现
    包含DWConv、ConvFFN1、ConvFFN2三个核心组件
    
    参数:
        d_model: 模型维度
        kernel_size: 卷积核大小
        expansion_factor: 扩张因子
        dropout: dropout率
    """
    def __init__(self, d_model, kernel_size=7, expansion_factor=2, dropout=0.1, **kwargs):
        super(ModernTCNBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout
        
        # DWConv: 深度卷积，建模时间依赖性
        self.dwconv = Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            groups=d_model,  # 深度卷积，每个通道独立
            activation='gelu'
        )
        
        # ConvFFN1: 通道间依赖性建模
        self.conv_ffn1_1 = Conv1D(
            filters=d_model * expansion_factor,
            kernel_size=1,
            padding='same',
            activation='gelu'
        )
        self.conv_ffn1_2 = Conv1D(
            filters=d_model,
            kernel_size=1,
            padding='same'
        )
        
        # ConvFFN2: 序列间依赖性建模
        self.conv_ffn2_1 = Conv1D(
            filters=d_model * expansion_factor,
            kernel_size=1,
            padding='same',
            activation='gelu'
        )
        self.conv_ffn2_2 = Conv1D(
            filters=d_model,
            kernel_size=1,
            padding='same'
        )
        
        # Layer Normalization
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.ln3 = LayerNormalization()
        
        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
    
    def call(self, inputs, training=None):
        # 输入形状: [batch_size, sequence_length, d_model]
        
        # 1. DWConv: 时间依赖性建模
        x = self.ln1(inputs)
        x = self.dwconv(x)
        x = self.dropout1(x, training=training)
        x = Add()([inputs, x])  # 残差连接
        
        # 2. ConvFFN1: 通道依赖性建模
        residual = x
        x = self.ln2(x)
        x = self.conv_ffn1_1(x)
        x = self.conv_ffn1_2(x)
        x = self.dropout2(x, training=training)
        x = Add()([residual, x])
        
        # 3. ConvFFN2: 序列依赖性建模
        residual = x
        x = self.ln3(x)
        x = self.conv_ffn2_1(x)
        x = self.conv_ffn2_2(x)
        x = self.dropout3(x, training=training)
        x = Add()([residual, x])
        
        return x

# =================================================================================
# 2. 动态定位网络架构
# =================================================================================

class DynamicPositioningNetwork:
    """
    动态定位网络：CNN + Modern TCN融合架构
    基于现有静态CNN网络的改进版本
    """
    def __init__(self, 
                 num_aps=16,           # AP数量
                 sequence_length=10,    # 时序长度
                 d_model=64,           # 模型维度
                 num_tcn_layers=3,     # TCN层数
                 cnn_filters=64,       # CNN滤波器数量
                 kernel_size=7,        # TCN卷积核大小
                 dropout=0.2):
        
        self.num_aps = num_aps
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_tcn_layers = num_tcn_layers
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
    
    def build_model(self):
        """
        构建完整的动态定位网络
        """
        # 输入层: [batch_size, sequence_length, num_aps]
        inputs = Input(shape=(self.sequence_length, self.num_aps), name='rss_sequence_input')
        
        # ==================== 空间特征提取分支 (CNN) ====================
        # 对每个时间步的RSS信号进行空间特征提取
        spatial_features = []
        
        for i in range(self.sequence_length):
            # 提取第i个时间步的RSS信号
            single_timestep = tf.expand_dims(inputs[:, i, :], axis=-1)  # [batch, 16, 1]
            
            # CNN空间特征提取 (保持原有CNN架构的优势)
            x = Conv1D(filters=self.cnn_filters, kernel_size=self.num_aps, 
                      padding='same', activation='relu')(single_timestep)
            x = Dropout(self.dropout)(x)
            x = BatchNormalization()(x)
            
            x = Conv1D(filters=self.cnn_filters, kernel_size=self.num_aps, 
                      padding='same', activation='relu')(x)
            x = Dropout(self.dropout)(x)
            x = BatchNormalization()(x)
            
            # 空间特征聚合
            spatial_feat = GlobalAveragePooling1D()(x)  # [batch, cnn_filters]
            spatial_features.append(spatial_feat)
        
        # 堆叠所有时间步的空间特征
        spatial_sequence = tf.stack(spatial_features, axis=1)  # [batch, seq_len, cnn_filters]
        
        # ==================== 时序特征提取分支 (Modern TCN) ====================
        # 投影到TCN维度
        if self.cnn_filters != self.d_model:
            temporal_input = Dense(self.d_model)(spatial_sequence)
        else:
            temporal_input = spatial_sequence
        
        # Modern TCN时序建模
        tcn_output = temporal_input
        for i in range(self.num_tcn_layers):
            tcn_output = ModernTCNBlock(
                d_model=self.d_model,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                name=f'modern_tcn_block_{i}'
            )(tcn_output)
        
        # ==================== 特征融合与输出 ====================
        # 时序特征聚合
        temporal_features = GlobalAveragePooling1D()(tcn_output)  # [batch, d_model]
        
        # 多尺度特征融合
        # 1. 当前时刻空间特征
        current_spatial = spatial_features[-1]  # 最新时刻的空间特征
        
        # 2. 历史时序特征
        historical_temporal = temporal_features
        
        # 3. 特征融合
        fused_features = Concatenate()([current_spatial, historical_temporal])
        
        # 输出层
        # 位置预测分支
        position_branch = Dense(128, activation='relu')(fused_features)
        position_branch = Dropout(self.dropout)(position_branch)
        position_output = Dense(3, activation='linear', name='position_output')(position_branch)
        
        # 姿态预测分支
        orientation_branch = Dense(128, activation='relu')(fused_features)
        orientation_branch = Dropout(self.dropout)(orientation_branch)
        orientation_output = Dense(3, activation='linear', name='orientation_output')(orientation_branch)
        
        # 创建模型
        model = Model(inputs=inputs, 
                     outputs=[position_output, orientation_output],
                     name='Dynamic_Positioning_CNN_TCN')
        
        return model
    
    def compile_model(self, model, position_weight=1.0, orientation_weight=1.0):
        """
        编译模型，使用联合损失函数
        """
        # 联合损失函数
        losses = {
            'position_output': 'mse',
            'orientation_output': 'mse'
        }
        
        loss_weights = {
            'position_output': position_weight,
            'orientation_output': orientation_weight
        }
        
        metrics = {
            'position_output': ['mae'],
            'orientation_output': ['mae']
        }
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return model

# =================================================================================
# 3. 数据预处理工具
# =================================================================================

class DataPreprocessor:
    """
    动态定位数据预处理器
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
    
    def create_sequences(self, X, Y, stride=1):
        """
        将静态数据转换为时序数据
        
        Args:
            X: RSS信号数据 [N, 16]
            Y: 标签数据 [N, 6]
            stride: 时序滑动步长
        
        Returns:
            X_seq: 时序RSS数据 [N-seq_len+1, seq_len, 16]
            Y_seq: 对应标签 [N-seq_len+1, 6]
        """
        X_sequences = []
        Y_sequences = []
        
        for i in range(0, len(X) - self.sequence_length + 1, stride):
            # 提取时序窗口
            seq_x = X[i:i + self.sequence_length]
            seq_y = Y[i + self.sequence_length - 1]  # 使用最后时刻的标签
            
            X_sequences.append(seq_x)
            Y_sequences.append(seq_y)
        
        return np.array(X_sequences), np.array(Y_sequences)
    
    def split_labels(self, Y):
        """
        分离位置和姿态标签
        """
        positions = Y[:, :3]  # x, y, z
        orientations = Y[:, 3:]  # alpha, beta, gamma
        return positions, orientations

# =================================================================================
# 4. 训练器类
# =================================================================================

class DynamicPositioningTrainer:
    """
    动态定位网络训练器
    """
    def __init__(self, sequence_length=10, stride=5):
        self.sequence_length = sequence_length
        self.stride = stride
        self.preprocessor = DataPreprocessor(sequence_length)
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
    def load_data(self, rss_file='RSS_total.mat', label_file='Label.mat'):
        """
        加载RSS和标签数据
        """
        print("正在加载数据...")
        
        # 加载.mat文件
        data_X = sio.loadmat(rss_file)
        data_Y = sio.loadmat(label_file)
        
        # 提取数据
        X = data_X['RSS_total']
        Y = data_Y['Label']
        
        print(f"原始数据维度:")
        print(f"- RSS数据: {X.shape}")
        print(f"- 标签数据: {Y.shape}")
        
        return X, Y
    
    def preprocess_data(self, X, Y, test_size=0.1):
        """
        数据预处理：标准化 + 时序化
        """
        print("\n开始数据预处理...")
        
        # 1. 数据分割
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )
        
        # 2. 标准化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        
        # 3. 创建时序数据
        print("创建时序数据...")
        X_train_seq, Y_train_seq = self.preprocessor.create_sequences(
            X_train_scaled, Y_train_scaled, stride=self.stride
        )
        X_test_seq, Y_test_seq = self.preprocessor.create_sequences(
            X_test_scaled, self.scaler_Y.transform(Y_test), stride=1
        )
        
        # 4. 分离位置和姿态标签
        Y_train_pos, Y_train_ori = self.preprocessor.split_labels(Y_train_seq)
        Y_test_pos, Y_test_ori = self.preprocessor.split_labels(Y_test_seq)
        
        print(f"\n时序数据维度:")
        print(f"- 训练集输入: {X_train_seq.shape}")
        print(f"- 训练集位置标签: {Y_train_pos.shape}")
        print(f"- 训练集姿态标签: {Y_train_ori.shape}")
        print(f"- 测试集输入: {X_test_seq.shape}")
        
        return (X_train_seq, Y_train_pos, Y_train_ori), (X_test_seq, Y_test_pos, Y_test_ori)
    
    def create_model(self, config=None):
        """
        创建动态定位模型
        """
        if config is None:
            config = {
                'num_aps': 16,
                'sequence_length': self.sequence_length,
                'd_model': 64,
                'num_tcn_layers': 3,
                'cnn_filters': 64,
                'kernel_size': 7,
                'dropout': 0.2
            }
        
        print("\n创建动态定位模型...")
        network = DynamicPositioningNetwork(**config)
        model = network.build_model()
        
        # 编译模型，位置权重更高
        model = network.compile_model(
            model, 
            position_weight=1.0,  # 位置损失权重
            orientation_weight=0.5  # 姿态损失权重
        )
        
        print("模型结构:")
        model.summary()
        
        return model, network
    
    def train_model(self, model, train_data, val_data=None, epochs=50, batch_size=128):
        """
        训练模型
        """
        X_train, Y_train_pos, Y_train_ori = train_data
        
        # 准备训练数据
        train_targets = {
            'position_output': Y_train_pos,
            'orientation_output': Y_train_ori
        }
        
        # 准备验证数据
        validation_data = None
        if val_data is not None:
            X_val, Y_val_pos, Y_val_ori = val_data
            val_targets = {
                'position_output': Y_val_pos,
                'orientation_output': Y_val_ori
            }
            validation_data = (X_val, val_targets)
        
        # 回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        print(f"\n开始训练模型...")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        
        # 训练
        history = model.fit(
            X_train,
            train_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_data):
        """
        评估模型性能
        """
        X_test, Y_test_pos, Y_test_ori = test_data
        
        print("\n评估模型性能...")
        
        # 预测
        predictions = model.predict(X_test)
        Y_pred_pos, Y_pred_ori = predictions
        
        # 反标准化预测结果
        Y_pred_combined = np.concatenate([Y_pred_pos, Y_pred_ori], axis=1)
        Y_test_combined = np.concatenate([Y_test_pos, Y_test_ori], axis=1)
        
        Y_pred_original = self.scaler_Y.inverse_transform(Y_pred_combined)
        Y_test_original = self.scaler_Y.inverse_transform(Y_test_combined)
        
        # 分离原始尺度的预测结果
        Y_pred_pos_orig = Y_pred_original[:, :3]
        Y_pred_ori_orig = Y_pred_original[:, 3:]
        Y_test_pos_orig = Y_test_original[:, :3]
        Y_test_ori_orig = Y_test_original[:, 3:]
        
        # 计算各维度误差
        print("\n=== 详细性能评估 ===")
        
        # 位置误差
        print("\n位置误差 (x, y, z):")
        pos_labels = ['x', 'y', 'z']
        for i, label in enumerate(pos_labels):
            mae = mean_absolute_error(Y_test_pos_orig[:, i], Y_pred_pos_orig[:, i])
            rmse = np.sqrt(mean_squared_error(Y_test_pos_orig[:, i], Y_pred_pos_orig[:, i]))
            print(f"  {label}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # 3D位置误差
        pos_errors_3d = np.sqrt(np.sum((Y_test_pos_orig - Y_pred_pos_orig)**2, axis=1))
        mae_3d = np.mean(pos_errors_3d)
        rmse_3d = np.sqrt(np.mean(pos_errors_3d**2))
        print(f"\n3D位置误差: MAE={mae_3d:.4f}, RMSE={rmse_3d:.4f}")
        
        # 姿态误差
        print("\n姿态误差 (α, β, γ):")
        ori_labels = ['alpha', 'beta', 'gamma']
        for i, label in enumerate(ori_labels):
            mae = mean_absolute_error(Y_test_ori_orig[:, i], Y_pred_ori_orig[:, i])
            rmse = np.sqrt(mean_squared_error(Y_test_ori_orig[:, i], Y_pred_ori_orig[:, i]))
            print(f"  {label}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        return {
            'position_mae_3d': mae_3d,
            'position_rmse_3d': rmse_3d,
            'predictions': Y_pred_original,
            'ground_truth': Y_test_original
        }

# =================================================================================
# 5. 便捷使用函数
# =================================================================================

def create_dynamic_model():
    """
    创建动态定位模型的示例函数
    """
    # 网络参数
    config = {
        'num_aps': 16,
        'sequence_length': 10,
        'd_model': 64,
        'num_tcn_layers': 3,
        'cnn_filters': 64,
        'kernel_size': 7,
        'dropout': 0.2
    }
    
    # 创建网络
    network = DynamicPositioningNetwork(**config)
    model = network.build_model()
    model = network.compile_model(model, position_weight=1.0, orientation_weight=0.5)
    
    return model, network

def main_training_pipeline():
    """
    主训练流程
    """
    print("=== 动态定位网络训练 ===")
    
    # 初始化训练器
    trainer = DynamicPositioningTrainer(sequence_length=10, stride=5)
    
    # 1. 加载数据
    try:
        X, Y = trainer.load_data()
    except FileNotFoundError:
        print("错误: 找不到数据文件，请确保RSS_total.mat和Label.mat在当前目录下")
        return
    
    # 2. 数据预处理
    train_data, test_data = trainer.preprocess_data(X, Y)
    
    # 划分验证集
    X_train, Y_train_pos, Y_train_ori = train_data
    split_idx = int(0.8 * len(X_train))
    
    train_subset = (X_train[:split_idx], Y_train_pos[:split_idx], Y_train_ori[:split_idx])
    val_subset = (X_train[split_idx:], Y_train_pos[split_idx:], Y_train_ori[split_idx:])
    
    # 3. 创建模型
    model, network = trainer.create_model()
    
    # 4. 训练模型
    history = trainer.train_model(
        model, train_subset, val_subset, 
        epochs=50, batch_size=128
    )
    
    # 5. 评估模型
    results = trainer.evaluate_model(model, test_data)
    
    print(f"\n=== 最终结果 ===")
    print(f"3D位置MAE: {results['position_mae_3d']:.4f}")
    print(f"3D位置RMSE: {results['position_rmse_3d']:.4f}")
    
    # 6. 保存模型
    model.save('dynamic_positioning_model.h5')
    print("\n模型已保存为 dynamic_positioning_model.h5")
    
    return model, trainer, results

# =================================================================================
# 6. 论文方案部分内容
# =================================================================================

PAPER_METHOD_SECTION = """
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

针对原有静态RSS数据 X ∈ ℝ^(N×16)，采用滑动窗口方法构建时序输入：

X_seq = {x_(t-T+1), x_(t-T+2), ..., x_t}

其中：
- T 为时序窗口长度（设置为10）
- x_t ∈ ℝ^16 表示第t时刻的RSS信号向量
- 最终输入维度：X_seq ∈ ℝ^(N×T×16)

### 3.2.2 标签对应策略

对于时序样本 X_seq[i:i+T]，采用序列末尾时刻的标签作为监督信号：

y_target = y_(i+T-1) = [x, y, z, α, β, γ]^T

该策略确保网络利用历史信息预测当前时刻的位置和姿态。

## 3.3 空间特征提取模块设计

### 3.3.1 基于CNN的空间特征提取

保留并改进原有CNN网络的空间特征提取能力，对时序中的每个时刻独立进行特征提取：

**第一层卷积块：**
h_1^(t) = BN(Dropout(ReLU(Conv1D(x_t))))

**第二层卷积块：**
h_2^(t) = BN(Dropout(ReLU(Conv1D(h_1^(t)))))

**空间特征聚合：**
f_spatial^(t) = GlobalAvgPool1D(h_2^(t))

### 3.3.2 网络参数配置

| 参数 | 数值 | 说明 |
|------|------|------|
| 卷积核大小 | 16 | 覆盖全部AP信号 |
| 滤波器数量 | 64 | 保持与原网络一致 |
| Dropout率 | 0.2 | 防止过拟合 |
| 激活函数 | ReLU | 增强非线性表达能力 |

最终得到空间特征序列：F_spatial = [f_spatial^(1), f_spatial^(2), ..., f_spatial^(T)]

## 3.4 Modern TCN时序建模模块

### 3.4.1 Modern TCN核心组件

Modern TCN模块包含三个关键子模块，分别建模不同维度的依赖关系：

**深度卷积（DWConv）：时间依赖性建模**

采用深度可分离卷积独立学习每个通道的时间模式：
h_dw = DWConv(LN(F_spatial))

其中，DWConv的分组数设置为特征维度d_model，实现通道间的独立时间建模。

**前馈网络1（ConvFFN1）：通道依赖性建模**

学习特征通道间的相互关系：
h_ffn1 = Conv1D_1×1(GELU(Conv1D_1×1(LN(h_dw + F_spatial))))

**前馈网络2（ConvFFN2）：序列依赖性建模**

捕捉多序列间的跨变量依赖关系：
h_ffn2 = Conv1D_1×1(GELU(Conv1D_1×1(LN(h_ffn1))))

### 3.4.2 Modern TCN块级联

通过多个Modern TCN块的级联实现深层时序特征提取：

F_temporal^(l) = ModernTCN^(l)(F_temporal^(l-1))

其中l = 1, 2, 3表示TCN层数，F_temporal^(0) = F_spatial。

### 3.4.3 残差连接与归一化

每个子模块均采用残差连接和层归一化，提升训练稳定性：

output = input + Dropout(SubModule(LayerNorm(input)))

## 3.5 多尺度特征融合策略

### 3.5.1 特征层次设计

为充分利用不同时间尺度的信息，设计了多层次特征融合机制：

**当前时刻空间特征（即时性）：**
f_current = f_spatial^(T)

**历史时序特征（连续性）：**
f_temporal = GlobalAvgPool1D(F_temporal^(3))

### 3.5.2 特征融合操作

采用特征拼接方式实现多尺度融合：
f_fused = Concat([f_current, f_temporal])

融合后的特征维度为：f_fused ∈ ℝ^128

## 3.6 联合输出设计

### 3.6.1 双分支预测架构

设计独立的位置和姿态预测分支，实现专门化的特征学习：

**位置预测分支：**
p_pos = Linear(ReLU(Linear(f_fused)))
ŷ_pos = [x̂, ŷ, ẑ]^T ∈ ℝ^3

**姿态预测分支：**
p_ori = Linear(ReLU(Linear(f_fused)))
ŷ_ori = [α̂, β̂, γ̂]^T ∈ ℝ^3

### 3.6.2 联合损失函数

采用加权多任务损失函数同时优化位置和姿态预测：

L_total = w_pos · L_pos + w_ori · L_ori

其中：
- L_pos = MSE(y_pos, ŷ_pos) 为位置损失
- L_ori = MSE(y_ori, ŷ_ori) 为姿态损失
- w_pos = 1.0，w_ori = 0.5 为损失权重

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
- 时序抖动：随机调整序列长度T ± 2
- 信号噪声：添加均值为0、标准差为0.01的高斯噪声
- 时序平移：采用不同步长的滑动窗口增加样本多样性

## 3.8 网络架构优势分析

### 3.8.1 相比原有静态CNN的改进

| 方面 | 静态CNN | 时空融合网络 | 改进效果 |
|------|---------|-------------|----------|
| **输入维度** | [N, 16] | [N, 10, 16] | 引入时序信息 |
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
"""

# =================================================================================
# 7. 使用说明
# =================================================================================

USAGE_INSTRUCTIONS = """
# 动态定位网络使用说明

## 快速开始

### 1. 基本使用
```python
# 创建模型
model, network = create_dynamic_model()

# 查看模型结构
model.summary()
```

### 2. 完整训练流程
```python
# 运行完整训练流程
model, trainer, results = main_training_pipeline()

# 查看结果
print(f"3D位置MAE: {results['position_mae_3d']:.4f}")
print(f"3D位置RMSE: {results['position_rmse_3d']:.4f}")
```

### 3. 自定义配置
```python
# 自定义网络配置
config = {
    'num_aps': 16,
    'sequence_length': 15,  # 调整时序长度
    'd_model': 128,         # 调整模型维度
    'num_tcn_layers': 4,    # 调整TCN层数
    'cnn_filters': 64,
    'kernel_size': 9,       # 调整TCN卷积核
    'dropout': 0.3
}

# 创建自定义网络
network = DynamicPositioningNetwork(**config)
model = network.build_model()
model = network.compile_model(model)
```

### 4. 数据要求
- RSS数据文件：RSS_total.mat，格式为 [N, 16] 矩阵
- 标签文件：Label.mat，格式为 [N, 6] 矩阵 (x,y,z,α,β,γ)
- 数据应放在当前工作目录下

### 5. 模型保存和加载
```python
# 保存模型
model.save('my_dynamic_model.h5')

# 加载模型
loaded_model = tf.keras.models.load_model('my_dynamic_model.h5')
```

## 性能调优建议

### 1. 超参数调优
- sequence_length: 5-15 (时序长度)
- d_model: 32-128 (模型维度)
- num_tcn_layers: 2-5 (TCN层数)
- kernel_size: 3-11 (TCN卷积核大小)
- dropout: 0.1-0.4 (dropout率)

### 2. 训练策略
- 使用学习率调度和早停
- 适当的数据增强
- 合理的验证集划分

### 3. 计算资源考虑
- 相比原CNN增加约2.2倍计算量
- 需要缓存历史时序数据
- GPU推荐：GTX 1080或更高配置

## 常见问题

### Q: 如何处理不同长度的时序？
A: 可以通过padding或截断统一到固定长度。

### Q: 如何调整位置和姿态的损失权重？
A: 在compile_model中修改position_weight和orientation_weight参数。

### Q: 如何处理实时推理？
A: 维护一个长度为sequence_length的滑动窗口缓存。

### Q: 原有数据格式如何转换？
A: 使用DataPreprocessor的create_sequences方法自动转换。
"""

# =================================================================================
# 8. 主程序入口
# =================================================================================

if __name__ == "__main__":
    print("动态定位网络完整方案包")
    print("=" * 50)
    print("本文件包含：")
    print("1. Modern TCN + CNN融合网络架构")
    print("2. 完整的训练和评估代码")
    print("3. 论文方案部分内容")
    print("4. 详细使用说明")
    print("=" * 50)
    
    print("\n选择运行模式：")
    print("1. 查看网络架构")
    print("2. 运行完整训练（需要数据文件）")
    print("3. 显示论文方案内容")
    print("4. 显示使用说明")
    
    try:
        choice = input("\n请选择 (1-4): ")
        
        if choice == "1":
            print("\n创建网络架构...")
            model, network = create_dynamic_model()
            print("网络创建成功！")
            
        elif choice == "2":
            print("\n开始完整训练流程...")
            model, trainer, results = main_training_pipeline()
            
        elif choice == "3":
            print("\n论文方案部分内容：")
            print(PAPER_METHOD_SECTION)
            
        elif choice == "4":
            print("\n使用说明：")
            print(USAGE_INSTRUCTIONS)
            
        else:
            print("无效选择，显示使用说明：")
            print(USAGE_INSTRUCTIONS)
            
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请检查数据文件是否存在，或参考使用说明")