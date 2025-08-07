import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Flatten, Dropout, BatchNormalization,
    Reshape, Concatenate, Add, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
import numpy as np

class ModernTCNBlock(tf.keras.layers.Layer):
    """
    Modern TCN模块实现
    包含DWConv、ConvFFN1、ConvFFN2三个核心组件
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
        
        # ConvFFN1: 通道间依赖性建模 (group=M)
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
        
        # ConvFFN2: 序列间依赖性建模 (group=D)
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

class DynamicPositioningNetwork:
    """
    动态定位网络：CNN + Modern TCN融合架构
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

# ==================== 数据预处理工具 ====================
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

# ==================== 使用示例 ====================
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

if __name__ == "__main__":
    # 创建模型
    model, network = create_dynamic_model()
    
    # 打印模型结构
    model.summary()
    
    # 可视化模型架构（可选）
    # tf.keras.utils.plot_model(model, to_file='dynamic_positioning_model.png', 
    #                          show_shapes=True, show_layer_names=True)