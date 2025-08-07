import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from Dynamic_CNN_TCN_Architecture import DynamicPositioningNetwork, DataPreprocessor, create_dynamic_model

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
    
    def plot_training_history(self, history):
        """
        绘制训练历史
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # 位置损失
        axes[0, 1].plot(history.history['position_output_loss'], label='Position Training Loss')
        axes[0, 1].plot(history.history['val_position_output_loss'], label='Position Validation Loss')
        axes[0, 1].set_title('Position Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # 姿态损失
        axes[1, 0].plot(history.history['orientation_output_loss'], label='Orientation Training Loss')
        axes[1, 0].plot(history.history['val_orientation_output_loss'], label='Orientation Validation Loss')
        axes[1, 0].set_title('Orientation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # 学习率
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
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
    
    # 6. 可视化训练历史
    trainer.plot_training_history(history)
    
    # 7. 保存模型
    model.save('dynamic_positioning_model.h5')
    print("\n模型已保存为 dynamic_positioning_model.h5")
    
    return model, trainer, results

if __name__ == "__main__":
    model, trainer, results = main()