import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用计算设备: {device}")
lasso_init_lgB = 19.7971
lasso_init_BoverC = -1.8582
lasso_init_ABC = -1.2986
lasso_init_intercept = 21.617
df = pd.read_csv(r'D:\damangshe\75\逆向PINN\ABC3_transformed.csv')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
target_col = df.columns[0]
feature_cols = df.columns[1:].tolist()
print(f"目标变量: {target_col}, 特征变量: {feature_cols}")
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=64, sigma=3.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, embed_dim // 2) * sigma, requires_grad=False)
    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], dim=-1)
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(in_dim + hidden_dim * 2, out_dim)
        self.activation = nn.Tanh()
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        out1 = self.activation(self.fc1(x))
        out2 = self.activation(self.fc2(torch.cat([x, out1], dim=-1)))
        out3 = self.fc3(torch.cat([x, out1, out2], dim=-1))
        return self.activation(out3 + self.residual(x))
class SolubilityPINN(nn.Module):
    def __init__(self, input_dim=3, fourier_embed_dim=64, hidden_dim=128):
        super().__init__()
        self.fourier_embed = FourierFeatureEmbedding(input_dim, fourier_embed_dim)
        embed_out_dim = fourier_embed_dim + input_dim
        self.res_blocks = nn.ModuleList([
            ResidualDenseBlock(embed_out_dim, hidden_dim, hidden_dim),
            ResidualDenseBlock(hidden_dim, hidden_dim, hidden_dim)
        ])
        self.fc_coeffs = nn.Linear(hidden_dim, 3)
        self.fc_coeffs.bias.data = torch.tensor([lasso_init_lgB, lasso_init_BoverC, lasso_init_ABC], dtype=torch.float32)
        self.constant_term = nn.Parameter(torch.tensor(lasso_init_intercept, dtype=torch.float32))
        self.fc_residual = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x_embed = self.fourier_embed(x)
        out = x_embed
        for block in self.res_blocks:
            out = block(out)
        coeffs = self.fc_coeffs(out)
        sol_prior = self.compute_sol_from_coeffs(x, coeffs)
        residual = self.fc_residual(out)
        sol_final = sol_prior + residual
        return sol_final, sol_prior, residual, coeffs
    def compute_sol_from_coeffs(self, x_norm, dynamic_coeffs):
        x_min = torch.tensor(scaler_X.data_min_, device=device, dtype=torch.float32)
        x_max = torch.tensor(scaler_X.data_max_, device=device, dtype=torch.float32)
        x_ori = x_norm * (x_max - x_min) + x_min
        lg_B = x_ori[:, 0:1]
        B_C = x_ori[:, 1:2]
        ABC = x_ori[:, 2:3]
        a = dynamic_coeffs[:, 0:1]
        b = dynamic_coeffs[:, 1:2]
        c = dynamic_coeffs[:, 2:3]
        sol_ori = a * lg_B + b * B_C + c * ABC + self.constant_term
        y_min = torch.tensor(scaler_y.data_min_, device=device, dtype=torch.float32)
        y_max = torch.tensor(scaler_y.data_max_, device=device, dtype=torch.float32)
        sol_norm = (sol_ori - y_min) / (y_max - y_min)
        return sol_norm
class LBFGSWrapper:
    def __init__(self, model, loss_fn, X_train, y_train, alpha=0.3, beta=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha  
        self.beta = beta    
        self.params = list(model.parameters()) 
        return torch.cat([p.reshape(-1) for p in self.params])
    def _unflatten_params(self, flat_params):
        idx = 0
        for p in self.params:
            param_size = p.numel()
            p.data = flat_params[idx:idx + param_size].view(p.shape)
            idx += param_size
    def _compute_loss_and_grad(self, flat_params):
        self._unflatten_params(torch.tensor(flat_params, dtype=torch.float32, device=device))
        self.model.zero_grad()
        sol_final, sol_prior, residual, _ = self.model(self.X_train)
        loss_main = self.loss_fn(sol_final, self.y_train)        
        loss_prior = self.loss_fn(sol_prior, self.y_train)       
        loss_residual_l2 = torch.mean(residual ** 2)            
        total_loss = loss_main + self.alpha * loss_prior + self.beta * loss_residual_l2
        total_loss.backward()
        grad = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p.reshape(-1))
                          for p in self.params]).detach().cpu().numpy()
        return total_loss.item(), grad
    def optimize(self, max_iter=80):
        initial_params = self._flatten_params().detach().cpu().numpy()
        result = opt.minimize(fun=self._compute_loss_and_grad, x0=initial_params,
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10, 'disp': False})
        self._unflatten_params(torch.tensor(result.x, dtype=torch.float32, device=device))
        return result
def train_single_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx,
                     alpha=0.3, beta=0.01, verbose=False):
    model = SolubilityPINN().to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    epochs = 1000
    best_val_r2 = -100
    best_model_path = f'best_pinn_model_fold_{fold_idx}.pth'
    X_train_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
    if verbose:
        print(f"\n--- 开始训练 Fold {fold_idx + 1} ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        sol_final, sol_prior, residual, _ = model(X_train_t)
        loss_main = mse_loss(sol_final, y_train_t)
        loss_prior = mse_loss(sol_prior, y_train_t)
        loss_residual_l2 = torch.mean(residual ** 2)
        total_loss = loss_main + alpha * loss_prior + beta * loss_residual_l2
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_final, _, _, _ = model(X_val_t)
            y_true_val = scaler_y.inverse_transform(y_val_t.cpu().numpy())
            val_sol_final = scaler_y.inverse_transform(val_final.cpu().numpy())
            val_r2 = r2_score(y_true_val, val_sol_final)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), best_model_path)
        if verbose and (epoch + 1) % 200 == 0:
            print(
                f"Fold {fold_idx + 1} Epoch [{epoch + 1:4d}/{epochs}] | Loss: {total_loss.item():.6f} | Val R2: {val_r2:.4f} | Best: {best_val_r2:.4f}")
        model.train()
    if verbose:
        print(f"--- Fold {fold_idx + 1} 开始L-BFGS精调 ---")
    model.load_state_dict(torch.load(best_model_path))
    lbfgs_optimizer = LBFGSWrapper(model, mse_loss, X_train_t, y_train_t, alpha=alpha, beta=beta)
    lbfgs_optimizer.optimize(max_iter=80)
    final_model_path = f'final_pinn_model_fold_{fold_idx}.pth'
    torch.save(model.state_dict(), final_model_path)
    return model, best_val_r2
def ensemble_predict(models, X_test_scaled):
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    preds_list = []
    for model in models:
        model.eval()
        with torch.no_grad():
            sol_final, _, _, _ = model(X_test_t)
            sol_pred = scaler_y.inverse_transform(sol_final.cpu().numpy())
            preds_list.append(sol_pred)
    ensemble_pred = np.mean(preds_list, axis=0)
    return ensemble_pred, preds_list
def evaluate_hyperparams(alpha, beta):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
    for fold_idx, (train_idx, _) in enumerate(kf.split(X_scaled)):
        X_train_fold_all = X_scaled[train_idx]
        y_train_fold_all = y_scaled[train_idx]
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold_all, y_train_fold_all, test_size=0.2, random_state=seed + fold_idx
        )
        _, best_val_r2 = train_single_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx,
            alpha=alpha, beta=beta, verbose=False
        )
        fold_scores.append(best_val_r2)
    mean_r2 = np.mean(fold_scores)
    std_r2 = np.std(fold_scores)
    return mean_r2, std_r2
if __name__ == "__main__":
    # ---------- 第一步：alpha × beta 超参数网格搜索 ----------
    print("\n" + "=" * 70)
    print("开始超参数网格搜索 (alpha × beta) | 三层损失函数版")
    print("=" * 70)
    alpha_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    beta_grid = [0.005, 0.01, 0.02, 0.05]
    results = []
    best_mean_r2 = -np.inf
    best_alpha = None
    best_beta = None
    total_combinations = len(alpha_grid) * len(beta_grid)
    current_idx = 0
    for alpha in alpha_grid:
        for beta in beta_grid:
            current_idx += 1
            print(f"\n[{current_idx}/{total_combinations}] 测试组合: alpha={alpha:.2f}, beta={beta:.3f}")
            mean_r2, std_r2 = evaluate_hyperparams(alpha, beta)
            results.append({
                'alpha': alpha,
                'beta': beta,
                'mean_val_R2': mean_r2,
                'std_val_R2': std_r2
            })
            print(f"  平均验证R²: {mean_r2:.4f} | 标准差: {std_r2:.4f}")
            if mean_r2 > best_mean_r2:
                best_mean_r2 = mean_r2
                best_alpha = alpha
                best_beta = beta
    print("\n" + "=" * 70)
    print("超参数网格搜索完成！结果排名（按R²降序）：")
    print("=" * 70)
    results_df = pd.DataFrame(results).sort_values('mean_val_R2', ascending=False).round(4)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 70)
    print(f"✅ 最优超参数: alpha = {best_alpha:.2f}, beta = {best_beta:.3f}")
    print(f"✅ 最优平均验证集R² = {best_mean_r2:.4f}")
    print("=" * 70)
    alpha = best_alpha
    beta = best_beta
    print("\n" + "=" * 70)
    print("使用最优超参数重新训练完整5折模型")
    print("=" * 70)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trained_models = []
    fold_val_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train_fold_all, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        y_train_fold_all, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold_all, y_train_fold_all, test_size=0.2, random_state=seed + fold_idx
        )

        print(f"\nFold {fold_idx + 1}/{n_splits} 数据划分:")
        print(f"  训练集: {len(X_train_fold)}, 验证集: {len(X_val_fold)}, 测试集: {len(X_test_fold)}")
        model, best_val_r2 = train_single_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx,
            alpha=alpha, beta=beta, verbose=True
        )
        trained_models.append(model)
        fold_val_scores.append(best_val_r2)
    print(f"\n{'=' * 70}")
    print(f"各折验证集最佳R2: {[f'{s:.4f}' for s in fold_val_scores]}")
    print(f"平均验证集R2: {np.mean(fold_val_scores):.4f}")
    print(f"{'=' * 70}")
    print(f"\n--- 开始集成评估 (在最后一折测试集上) ---")
    _, last_test_idx = list(kf.split(X_scaled))[-1]
    X_test_final = X_scaled[last_test_idx]
    y_test_final = y_scaled[last_test_idx]
    y_true_final = scaler_y.inverse_transform(y_test_final)
    ensemble_pred, individual_preds = ensemble_predict(trained_models, X_test_final)
    print(f"\n--- 各模型单模型在测试集上的表现 ---")
    for i, pred in enumerate(individual_preds):
        r2 = r2_score(y_true_final, pred)
        print(f"  Model {i + 1} R2: {r2:.6f}")

    print(f"\n--- 🎉 最优参数集成模型最终测试表现 ---")
    final_metrics = {
        'R2': r2_score(y_true_final, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true_final, ensemble_pred)),
        'MAE': mean_absolute_error(y_true_final, ensemble_pred)
    }
    print(f"==========================")
    print(f"测试集 R2   = {final_metrics['R2']:.6f}")
    print(f"测试集 RMSE = {final_metrics['RMSE']:.6f}")
    print(f"测试集 MAE  = {final_metrics['MAE']:.6f}")
    print(f"==========================")
    print("\n" + "=" * 70)
    print("全数据集平均偏导数 (GAPD) 分析")
    print("=" * 70)
    X_all_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).to(device)
    all_grads = []
    for model in trained_models:
        model.eval()
        with torch.enable_grad():
            sol_final_all, _, _, _ = model(X_all_tensor)
            grads = torch.autograd.grad(
                outputs=sol_final_all,
                inputs=X_all_tensor,
                grad_outputs=torch.ones_like(sol_final_all)
            )[0]
            all_grads.append(grads.detach().cpu().numpy())
    mean_grads = np.mean(all_grads, axis=0)
    mean_grad_lgB = np.mean(mean_grads[:, 0])
    mean_grad_BC = np.mean(mean_grads[:, 1])
    mean_grad_ABC = np.mean(mean_grads[:, 2])
    print("\n========== 全数据集 平均偏导数 (集成后) ==========")
    print(f"平均 dSOL/dlg_B    = {mean_grad_lgB:.6f}")
    print(f"平均 dSOL/d(B/C)   = {mean_grad_BC:.6f}")
    print(f"平均 dSOL/d(ABC)   = {mean_grad_ABC:.6f}")
    print("=====================================================\n")
    print("========== 训练后可训练常数项（5折均值） ==========")
    const_values = [model.constant_term.item() for model in trained_models]
    print(f"各折常数项: {[f'{v:.4f}' for v in const_values]}")
    print(f"平均常数项: {np.mean(const_values):.4f}")
    print("=====================================================\n")

    print("🎉 全部运行完成！三层损失函数版已验证！")
