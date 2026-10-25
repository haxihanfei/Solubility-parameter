
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import (
    train_test_split, RepeatedKFold  
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import shap  
tf.keras.backend.clear_session()
tf.random.set_seed(42)  
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  
print(f"当前SHAP版本：{shap.__version__}")  
print(f"当前TensorFlow版本：{tf.__version__}") 
def load_and_preprocess_data(file_path, for_cv=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在，请检查路径：{file_path}")
    df = pd.read_csv(file_path)
    print(f"\n正在加载数据: {file_path}")
    print(f"数据集原始信息：{df.shape[0]}个样本，{df.shape[1] - 1}个特征，1个目标变量")
    feature_molmr_name = "HS MolMR"  
    interaction_feature_name = None
    if df.isnull().sum().sum() > 0:
        missing_count = df.isnull().sum().sum()
        print(f"发现{missing_count}个缺失值，已自动填充为对应列的均值")
        df = df.fillna(df.mean())
    y = df.iloc[:, 0].values  
    X = df.iloc[:, 1:].values  
    feature_names = df.columns[1:].tolist() 
    scaler_X_global = StandardScaler()
    X_scaled_global = scaler_X_global.fit_transform(X)
    scaler_y_global = StandardScaler()
    if for_cv:
        return X, y, feature_names, feature_molmr_name, interaction_feature_name, scaler_X_global, scaler_y_global
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        print(f"训练集维度（适配1D-CNN）：{X_train_reshaped.shape}")
        print(f"测试集维度（适配1D-CNN）：{X_test_reshaped.shape}")
        return (X_train_reshaped, X_test_reshaped, y_train_scaled, y_test_scaled,
                scaler_X, scaler_y, feature_names, y_train, y_test, X_train_scaled, X_test_scaled,
                feature_molmr_name, interaction_feature_name)
def build_1d_cnn_model(input_shape):
    model = Sequential(name="1D_CNN_Regression_Model")
    model.add(Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=input_shape,
        name="Conv1D_1"
    ))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool_1"))
    model.add(Dropout(0.2, name="Dropout_1"))
    model.add(Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        name="Conv1D_2"
    ))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool_2"))
    model.add(Dropout(0.2, name="Dropout_2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(128, activation='relu', name="Dense_1"))
    model.add(Dropout(0.3, name="Dropout_3"))
    model.add(Dense(64, activation='relu', name="Dense_2"))
    model.add(Dropout(0.3, name="Dropout_4"))
    model.add(Dense(1, activation='linear', name="Output"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model
def run_3times_10fold_cv(X, y, feature_names, feature_molmr_name, interaction_feature_name):
    n_splits = 10  
    n_repeats = 3  
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )
    cv_metrics = {
        'r2': [],
        'rmse': [],
        'mae': [],
        'mse': []
    }
    cv_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=0,
            restore_best_weights=True
        )
    ]
    print("\n" + "=" * 80)
    print(f"开始执行 {n_repeats} 次 {n_splits} 折重复交叉验证...")
    print("=" * 80)
    fold_count = 0
    total_folds = n_splits * n_repeats
    for repeat_idx, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
        fold_count += 1
        repeat_num = repeat_idx // n_splits + 1
        fold_num = repeat_idx % n_splits + 1
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]
        scaler_X_cv = StandardScaler()
        X_cv_train_scaled = scaler_X_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_X_cv.transform(X_cv_val)
        scaler_y_cv = StandardScaler()
        y_cv_train_scaled = scaler_y_cv.fit_transform(y_cv_train.reshape(-1, 1)).flatten()
        y_cv_val_scaled = scaler_y_cv.transform(y_cv_val.reshape(-1, 1)).flatten()
        input_shape = (X_cv_train_scaled.shape[1], 1)
        X_cv_train_reshaped = X_cv_train_scaled.reshape(X_cv_train_scaled.shape[0], input_shape[0], 1)
        X_cv_val_reshaped = X_cv_val_scaled.reshape(X_cv_val_scaled.shape[0], input_shape[0], 1)
        tf.keras.backend.clear_session() 
        model_cv = build_1d_cnn_model(input_shape)
        model_cv.fit(
            X_cv_train_reshaped, y_cv_train_scaled,
            batch_size=32,
            epochs=50,
            validation_data=(X_cv_val_reshaped, y_cv_val_scaled),
            callbacks=cv_callbacks,
            verbose=0
        )
        y_cv_pred_scaled = model_cv.predict(X_cv_val_reshaped, verbose=0).flatten()
        y_cv_pred = scaler_y_cv.inverse_transform(y_cv_pred_scaled.reshape(-1, 1)).flatten()
        y_cv_true = scaler_y_cv.inverse_transform(y_cv_val_scaled.reshape(-1, 1)).flatten()
        r2 = r2_score(y_cv_true, y_cv_pred)
        mse = mean_squared_error(y_cv_true, y_cv_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_cv_true, y_cv_pred)
        cv_metrics['r2'].append(r2)
        cv_metrics['rmse'].append(rmse)
        cv_metrics['mae'].append(mae)
        cv_metrics['mse'].append(mse)
        if fold_count % 5 == 0 or fold_count == total_folds:
            print(
                f"完成 {fold_count}/{total_folds} 折叠 | 重复{repeat_num}次-折{fold_num} | 当前R²均值：{np.mean(cv_metrics['r2']):.4f}")
    cv_results = {
        'R²': {
            'mean': np.mean(cv_metrics['r2']),
            'std': np.std(cv_metrics['r2']),
            'all_values': cv_metrics['r2']
        },
        'RMSE': {
            'mean': np.mean(cv_metrics['rmse']),
            'std': np.std(cv_metrics['rmse'])
        },
        'MAE': {
            'mean': np.mean(cv_metrics['mae']),
            'std': np.std(cv_metrics['mae'])
        },
        'MSE': {
            'mean': np.mean(cv_metrics['mse']),
            'std': np.std(cv_metrics['mse'])
        }
    }
    print("\n" + "=" * 80)
    print(f"{n_repeats} 次 {n_splits} 折交叉验证最终结果")
    print("=" * 80)
    print(f"R²（决定系数）：{cv_results['R²']['mean']:.4f} ± {cv_results['R²']['std']:.4f}")
    print(f"RMSE（均方根误差）：{cv_results['RMSE']['mean']:.4f} ± {cv_results['RMSE']['std']:.4f}")
    print(f"MAE（平均绝对误差）：{cv_results['MAE']['mean']:.4f} ± {cv_results['MAE']['std']:.4f}")
    print(f"MSE（均方误差）：{cv_results['MSE']['mean']:.4f} ± {cv_results['MSE']['std']:.4f}")
    print("=" * 80)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.boxplot(cv_metrics['r2'], patch_artist=True, boxprops=dict(facecolor='#2E86AB'))
    plt.title(f'交叉验证R²分布（均值={cv_results["R²"]["mean"]:.4f}）')
    plt.ylabel('R²值')
    plt.grid(alpha=0.3, linestyle='--')
    plt.subplot(2, 2, 2)
    plt.boxplot(cv_metrics['rmse'], patch_artist=True, boxprops=dict(facecolor='#A23B72'))
    plt.title(f'交叉验证RMSE分布（均值={cv_results["RMSE"]["mean"]:.4f}）')
    plt.ylabel('RMSE值')
    plt.grid(alpha=0.3, linestyle='--')
    plt.subplot(2, 2, 3)
    plt.boxplot(cv_metrics['mae'], patch_artist=True, boxprops=dict(facecolor='#F18F01'))
    plt.title(f'交叉验证MAE分布（均值={cv_results["MAE"]["mean"]:.4f}）')
    plt.ylabel('MAE值')
    plt.grid(alpha=0.3, linestyle='--')
    plt.subplot(2, 2, 4)
    plt.boxplot(cv_metrics['mse'], patch_artist=True, boxprops=dict(facecolor='#C73E1D'))
    plt.title(f'交叉验证MSE分布（均值={cv_results["MSE"]["mean"]:.4f}）')
    plt.ylabel('MSE值')
    plt.grid(alpha=0.3, linestyle='--')
    plt.suptitle(f'{n_repeats}次{n_splits}折交叉验证指标分布', fontsize=16)
    plt.tight_layout()
    plt.show()
    return cv_results, cv_metrics
file_path = r"D:.csv" 
X_full, y_full, feature_names, feature_molmr_name, interaction_feature_name, _, _ = load_and_preprocess_data(
    file_path, for_cv=True
)
cv_results, cv_metrics = run_3times_10fold_cv(
    X_full, y_full, feature_names, feature_molmr_name, interaction_feature_name
)
(
    X_train, X_test, y_train_scaled, y_test_scaled,
    scaler_X, scaler_y, feature_names, y_train, y_test, X_train_scaled_flat, X_test_scaled_flat,
    feature_molmr_name, interaction_feature_name
) = load_and_preprocess_data(file_path, for_cv=False)
def train_model(model, X_train, y_train, X_test, y_test):
    """
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="1d_cnn_best_model.keras",
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    print("\n==================== 开始训练最终模型（原始稳定版） ====================")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    # 将history中的指标转换为DataFrame
    history_df = pd.DataFrame(history.history)
    # 添加epoch列（从1开始）
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
    # 保存到CSV文件
    csv_path = "training_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"\n训练历史数据（损失和MAE曲线原始数据）已保存至：{csv_path}")
    # ----------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失（MSE）', color='#2E86AB')
    plt.plot(history.history['val_loss'], label='验证损失（MSE）', color='#A23B72')
    plt.title('模型训练与验证损失曲线')
    plt.xlabel('训练轮数（Epoch）')
    plt.ylabel('损失值（MSE）')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='训练MAE', color='#2E86AB')
    plt.plot(history.history['val_mean_absolute_error'], label='验证MAE', color='#A23B72')
    plt.title('模型训练与验证MAE曲线')
    plt.xlabel('训练轮数（Epoch）')
    plt.ylabel('平均绝对误差（MAE）')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    return model, history
input_shape = (X_train.shape[1], X_train.shape[2])
final_model = build_1d_cnn_model(input_shape)
print("\n==================== 最终1D-CNN模型结构（原始稳定版） ====================")
final_model.summary()
trained_model, training_history = train_model(final_model, X_train, y_train_scaled, X_test, y_test_scaled)
def evaluate_model(model, X_test, y_test_scaled, scaler_y, y_test):
    """
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print("\n==================== 最终模型测试集评估结果 ====================")
    print(f"R²（决定系数）：{r2:.4f}")
    print(f"RMSE（均方根误差）：{rmse:.4f}")
    print(f"MAE（平均绝对误差）：{mae:.4f}")
    print(f"MSE（均方误差）：{mse:.4f}")
    print(f"\n对比交叉验证均值：")
    print(f"交叉验证R²均值：{cv_results['R²']['mean']:.4f}")
    print(f"交叉验证RMSE均值：{cv_results['RMSE']['mean']:.4f}")
    print(f"交叉验证MAE均值：{cv_results['MAE']['mean']:.4f}")
    print(f"交叉验证MSE均值：{cv_results['MSE']['mean']:.4f}")
    print("=" * 50)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#2E86AB', label='预测样本点')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线（y=x）')
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('1D-CNN模型：实际值 vs 预测值')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.show()
    return r2, rmse, mae, mse, y_pred
r2_score_result, rmse_result, mae_result, mse_result, y_pred = evaluate_model(
    trained_model, X_test, y_test_scaled, scaler_y, y_test
)
def shap_feature_analysis(model, X_train, X_test, feature_names, scaler_X,
                          X_train_scaled_flat, X_test_scaled_flat,
                          feature_molmr_name, interaction_feature_name):  # 接收关键名称（已无实际交互特征）
    """
    """
    try:
        background_data = X_train[:100]
        X_test_sample = X_test[:len(X_test)]
        X_test_sample_flat = X_test_scaled_flat[:len(X_test)]
        explainer = shap.DeepExplainer(
            model=model,
            data=background_data
        )
        shap_values = explainer.shap_values(X_test_sample)
        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_values_flat = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
        X_test_original_flat = scaler_X.inverse_transform(X_test_sample_flat)
        feature_importance = np.mean(np.abs(shap_values_flat), axis=0)
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'shap_importance': feature_importance
        }).sort_values(by='shap_importance', ascending=False).reset_index(drop=True)
        top_6_features = importance_df.head(6)
        print("\n前6大最重要特征（按SHAP重要性排序）：")
        print(top_6_features.to_string(index=False))
        plt.figure(figsize=(8, 8))
        bars = plt.barh(
            range(len(top_6_features)),
            top_6_features['shap_importance'],
            color='#A23B72'
        )
        for i, bar in enumerate(bars):
            bar_width = bar.get_width()
            plt.text(bar_width + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{bar_width:.4f}', ha='left', va='center', fontsize=9)
        plt.yticks(range(len(top_6_features)), top_6_features['feature_name'], fontsize=10)
        plt.xlabel('SHAP重要性（绝对值均值）', fontsize=12)
        plt.title('1D-CNN模型：前6大重要特征（SHAP分析）', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
        shap_expl = shap.Explanation(
            values=shap_values_flat,
            data=X_test_original_flat,
            feature_names=feature_names
        )
        plt.figure(figsize=(12, 7))
        shap.plots.beeswarm(
            shap_expl,
            show=False,
            color_bar_label="特征原始值大小",
            max_display=6  # 核心：固定展示前6大特征
        )
        plt.title('SHAP特征影响分布（蜂群图）- 前6大特征', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        return top_6_features
    except Exception as e:
        print(f"\nSHAP分析过程中出现错误：{str(e)}")
        print("建议排查：1. SHAP与TensorFlow版本兼容性 2. 数据维度是否匹配 3. HS MolMR列名是否正确")
        return None
shap_top_6_features = shap_feature_analysis(
    trained_model, X_train, X_test, feature_names, scaler_X,
    X_train_scaled_flat, X_test_scaled_flat,
    feature_molmr_name, interaction_feature_name
)
print("\n" + "=" * 80)
print("最终结果汇总")
print("=" * 80)
print("【3次10折交叉验证结果（均值±标准差）】")
print(f"R²：{cv_results['R²']['mean']:.4f} ± {cv_results['R²']['std']:.4f}")
print(f"RMSE：{cv_results['RMSE']['mean']:.4f} ± {cv_results['RMSE']['std']:.4f}")
print(f"MAE：{cv_results['MAE']['mean']:.4f} ± {cv_results['MAE']['std']:.4f}")
print(f"MSE：{cv_results['MSE']['mean']:.4f} ± {cv_results['MSE']['std']:.4f}")
print("\n【最终模型测试集结果】")
print(f"R²：{r2_score_result:.4f}")
print(f"RMSE：{rmse_result:.4f}")
print(f"MAE：{mae_result:.4f}")
print(f"MSE：{mse_result:.4f}")
print("=" * 80)
