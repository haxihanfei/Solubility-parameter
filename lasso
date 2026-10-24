import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 关闭收敛警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)
def load_preprocess_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 数据集不存在：{input_path}")
    if not input_path.endswith(".csv"):
        raise ValueError(f"❌ 仅支持CSV格式")
    df = pd.read_csv(input_path)
    print(f"✅ 成功读取数据：{input_path}")
    print(f"📊 数据规模：{df.shape[0]} 样本 × {df.shape[1]} 列")
    target_name = df.columns[0]
    feature_names = df.columns[1:].tolist()
    y = df[target_name].values
    X = df[feature_names].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"✅ 训练集：{X_train.shape[0]} 样本 | 测试集：{X_test.shape[0]} 样本")
    print(f"\n 目标变量：{target_name}")
    print(f"   特征列：{feature_names}")
    return X_train, X_test, y_train, y_test, target_name, feature_names, df
def lasso_regression_fit(X_train, X_test, y_train, y_test, target_name, feature_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso_cv = LassoCV(
        alphas=np.logspace(-6, 1, 50),
        cv=5,
        random_state=42,
        max_iter=50000,
        tol=1e-6,
        n_jobs=-1
    )
    lasso_cv.fit(X_train_scaled, y_train)
    best_alpha = lasso_cv.alpha_
    lasso_model = Lasso(
        alpha=best_alpha,
        fit_intercept=True,
        random_state=42,
        max_iter=50000,
        tol=1e-6
    )
    lasso_model.fit(X_train_scaled, y_train)
    y_train_pred = lasso_model.predict(X_train_scaled)
    y_test_pred = lasso_model.predict(X_test_scaled)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"\n📈 模型评估：")
    print(f"   最优alpha = {best_alpha:.6f}")
    print(f"   训练R²：{r2_train:.4f} | 测试R²：{r2_test:.4f}")
    print(f"   训练RMSE：{rmse_train:.4f} | 测试RMSE：{rmse_test:.4f}")
    print(f"   训练MAE：{mae_train:.4f} | 测试MAE：{mae_test:.4f}")  # 打印MAE
    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_
    threshold = 1e-6
    kept_mask = np.abs(coefficients) > threshold
    kept_feats = [f for f, keep in zip(feature_names, kept_mask) if keep]
    kept_coefs = [c for c, keep in zip(coefficients, kept_mask) if keep]
    kept_feat_count = len(kept_feats)
    dropped_feat_count = len(feature_names) - kept_feat_count
    print(f"\n 特征筛选：保留 {kept_feat_count} 个 | 剔除 {dropped_feat_count} 个")
    for f, c in zip(kept_feats, kept_coefs):
        print(f"   {f:<10} {c:.4f}")
    formula_parts = []
    for i, (feat_name, coef) in enumerate(zip(kept_feats, kept_coefs)):
        if i == 0:
            formula_parts.append(f"{coef:.4f} × {feat_name}")
        else:
            if coef >= 0:
                formula_parts.append(f" + {coef:.4f} × {feat_name}")
            else:
                formula_parts.append(f" - {abs(coef):.4f} × {feat_name}")
    if intercept >= 0:
        formula_parts.append(f" + {intercept:.4f}")
    else:
        formula_parts.append(f" - {abs(intercept):.4f}")
    quantitative_formula = f"{target_name} = " + "".join(formula_parts)
    print(f"\n📝 回归公式：\n   {quantitative_formula}")
    plt.figure()
    plt.scatter(y_train, y_train_pred, alpha=0.6, color="#2E86AB", label=f"训练 R²={r2_train:.4f}")
    plt.scatter(y_test, y_test_pred, alpha=0.8, color="#E74C3C", label=f"测试 R²={r2_test:.4f}")
    plt.plot([min(y_test.min(), y_train.min()), max(y_test.max(), y_train.max())],
             [min(y_test.min(), y_train.min()), max(y_test.max(), y_train.max())], 'k--', lw=2)
    plt.xlabel(f"实际{target_name}")
    plt.ylabel(f"预测{target_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(input_path), "Lasso拟合效果图.png"), bbox_inches='tight')
    plt.close()
    plt.figure()
    colors = ["#2E86AB" if keep else "#CCCCCC" for keep in kept_mask]
    plt.barh(feature_names, coefficients, color=colors)
    plt.axvline(0, c='k', ls='--', alpha=0.5)
    plt.xlabel("特征系数")
    plt.title(f"保留特征：{kept_feat_count} 个")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(input_path), "Lasso特征系数分布图.png"), bbox_inches='tight')
    plt.close()
    joblib.dump({"model": lasso_model, "scaler": scaler},
                os.path.join(os.path.dirname(input_path), "Lasso回归模型_最优alpha.pkl"))
    with open(os.path.join(os.path.dirname(input_path), "Lasso回归结果报告.txt"), "w", encoding="utf-8") as f:
        f.write(f"最优alpha：{best_alpha:.6f}\n")
        f.write(f"训练集 R²：{r2_train:.4f}\n")
        f.write(f"测试集 R²：{r2_test:.4f}\n")
        f.write(f"训练集 RMSE：{rmse_train:.4f}\n")
        f.write(f"测试集 RMSE：{rmse_test:.4f}\n")
        f.write(f"训练集 MAE：{mae_train:.4f}\n") 
        f.write(f"测试集 MAE：{mae_test:.4f}\n\n")
        f.write(f"保留特征：{kept_feats}\n\n")
        f.write(f"公式：{quantitative_formula}\n")
    print(f"\n✅ 文件已保存到：{os.path.dirname(input_path)}")
    return lasso_model, scaler, quantitative_formula, r2_train, r2_test, kept_feats, best_alpha
if __name__ == "__main__":
    input_path = r"D:.csv"

    try:
        X_train, X_test, y_train, y_test, target_name, feature_names, df = load_preprocess_data(input_path)
        trained_model, scaler, linear_formula, r2_train, r2_test, kept_feats, best_alpha = lasso_regression_fit(
            X_train, X_test, y_train, y_test, target_name, feature_names
        )
        print("\n" + "="*60)
        print("🎉 运行完成！R² / RMSE / MAE 已全部计算")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 错误：{e}")
