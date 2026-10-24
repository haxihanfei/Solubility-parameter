import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
CSV_FILE_PATH = r'D:.csv'
seed = 42
np.random.seed(seed)
if not os.path.exists(CSV_FILE_PATH):
    print(f"❌ 错误：找不到数据集文件")
    print(f"   查找路径：{CSV_FILE_PATH}")
    exit(1)
df = pd.read_csv(CSV_FILE_PATH)
print(f"成功读取数据集")
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
target_col = df.columns[0]
feature_cols = df.columns[1:].tolist()
print(f"目标变量: {target_col}, 特征变量: {feature_cols}")
print(f"特征数量: {len(feature_cols)}")
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
def evaluate_gpr_cv():
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = {
        'r2': [],
        'rmse': [],
        'mae': []
    }
    print("\n" + "=" * 70)
    print("高斯过程回归 5折交叉验证")
    print("=" * 70)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train_fold_all = X_scaled[train_idx]
        y_train_fold_all = y_scaled[train_idx]
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold_all, y_train_fold_all, test_size=0.2, random_state=seed + fold_idx
        )
        kernel = RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=seed
        )
        gpr.fit(X_train_fold, y_train_fold.ravel())
        y_val_pred_scaled = gpr.predict(X_val_fold)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
        y_val_true = scaler_y.inverse_transform(y_val_fold).flatten()
        r2 = r2_score(y_val_true, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
        mae = mean_absolute_error(y_val_true, y_val_pred)
        fold_scores['r2'].append(r2)
        fold_scores['rmse'].append(rmse)
        fold_scores['mae'].append(mae)
        print(f"Fold {fold_idx+1}/5 | 验证R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    cv_results = {
        'R²': {'mean': np.mean(fold_scores['r2']), 'std': np.std(fold_scores['r2'])},
        'RMSE': {'mean': np.mean(fold_scores['rmse']), 'std': np.std(fold_scores['rmse'])},
        'MAE': {'mean': np.mean(fold_scores['mae']), 'std': np.std(fold_scores['mae'])}
    }
    print(f"\n 5折交叉验证平均结果：")
    print(f"R²：{cv_results['R²']['mean']:.4f} ± {cv_results['R²']['std']:.4f}")
    print(f"RMSE：{cv_results['RMSE']['mean']:.4f} ± {cv_results['RMSE']['std']:.4f}")
    print(f"MAE：{cv_results['MAE']['mean']:.4f} ± {cv_results['MAE']['std']:.4f}")
    return cv_results, gpr
if __name__ == "__main__":
    cv_results, _ = evaluate_gpr_cv()
    print("\n" + "=" * 70)
    print("最终测试集评估")
    print("=" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    _, last_test_idx = list(kf.split(X_scaled))[-1]
    X_test_final = X_scaled[last_test_idx]
    y_test_final = y_scaled[last_test_idx]
    train_idx_all = list(kf.split(X_scaled))[-1][0]
    X_train_final_all = X_scaled[train_idx_all]
    y_train_final_all = y_scaled[train_idx_all]
    kernel_final = RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    final_gpr = GaussianProcessRegressor(
        kernel=kernel_final,
        n_restarts_optimizer=5,
        random_state=seed
    )
    final_gpr.fit(X_train_final_all, y_train_final_all.ravel())
    y_test_pred_scaled = final_gpr.predict(X_test_final)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_true = scaler_y.inverse_transform(y_test_final).flatten()
    test_r2 = r2_score(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    print(f"\n📊 高斯过程回归 测试集最终结果：")
    print(f"R²：{test_r2:.6f}")
    print(f"RMSE：{test_rmse:.6f}")
    print(f"MAE：{test_mae:.6f}")
    print("=" * 70)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(y_test_true, y_test_pred, alpha=0.7, color='#3399FF', s=15)
    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='理想预测线')
    plt.xlabel('真实溶解度', fontsize=12)
    plt.ylabel('预测溶解度', fontsize=12)
    plt.title(f'高斯过程回归：真实值 vs 预测值\n测试集R² = {test_r2:.4f}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('GPR测试集预测图.png', bbox_inches='tight')
    plt.close()
    print("\n✅ 测试集预测图已保存为 GPR测试集预测图.png")
    print("\n🎉 高斯过程回归模型全部运行完成！")
