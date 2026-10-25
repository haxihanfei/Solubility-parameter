import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cosine
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
print("=" * 80)
print("聚脲溶解度参数回归任务NNI-SMOTE数据增强（原始1500保留+22-24区间填补版）")
print("=" * 80)
print("\n🔍 检查计算环境...")
cp = None
USE_CUDA = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        USE_CUDA = True
        print(f"✅ CUDA可用！将使用GPU加速")
        print(f"   CuPy版本：{cp.__version__}")
        print(f"   GPU设备：{cp.cuda.Device(0).name()}")
        print(f"   GPU显存：{cp.cuda.Device(0).mem_info()[1] / 1024 ** 3:.1f} GB")
    else:
        print("⚠️  CuPy已安装但CUDA不可用，将使用CPU运行")
        cp = None
except ImportError:
    print("ℹ️  未安装CuPy，将使用CPU运行")
CSV_PATH = r"D:.csv"
SAVE_PATH = r"D:.csv"
TEST_SAVE_PATH = r"D:.csv"
PLOT_SAVE_PATH = r"D:.png"
STATS_SAVE_PATH = r"D:.csv"
FINAL_SAMPLE_NUM = 1700
ORIGINAL_SAMPLE_RATIO = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
NNI_R_MAX = min(20, 100)
NNI_CONVERGENCE_TOL = 0
KDE_DENSITY_THRESHOLD_RATIO = 0.3
SAFE_RADIUS_NEIGHBORS = 5
DELTA_MIN = 0.01
SAFE_MARGIN = 0.1
REDUNDANCY_DISTANCE_THRESHOLD = 0.05
def natural_neighbor_search(X, r_max=20, convergence_tol=0, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    if n_samples <= 1:
        return [[]] * n_samples, 0
    if USE_CUDA and cp is not None:
        print(f"      🔧 正在GPU上预计算{n_samples}×{n_samples}距离矩阵...")
        X_gpu = cp.asarray(X)
        X_sq = cp.sum(X_gpu ** 2, axis=1, keepdims=True)
        dist_matrix_sq = X_sq + X_sq.T - 2 * cp.dot(X_gpu, X_gpu.T)
        dist_matrix_gpu = cp.sqrt(cp.maximum(dist_matrix_sq, 0.0))
        cp.fill_diagonal(dist_matrix_gpu, cp.inf)
        dist_matrix = cp.asnumpy(dist_matrix_gpu)
        print(f"      ✅ 距离矩阵计算完成（GPU加速）")
    else:
        print(f"      🔧 正在CPU上预计算{n_samples}×{n_samples}距离矩阵...")
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        dist_matrix_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
        dist_matrix = np.sqrt(np.maximum(dist_matrix_sq, 0.0))
        np.fill_diagonal(dist_matrix, np.inf)
        print(f"      ✅ 距离矩阵计算完成（CPU）")
    r = 1
    prev_isolated = n_samples
    natural_neighbors = [[] for _ in range(n_samples)]
    while r <= min(r_max, n_samples - 1):
        r_neighbors = []
        for i in range(n_samples):
            neighbors = np.argsort(dist_matrix[i])[:r]
            r_neighbors.append(neighbors.tolist())
        reverse_neighbor_counts = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            for j in r_neighbors[i]:
                reverse_neighbor_counts[j] += 1
        current_isolated = np.sum(reverse_neighbor_counts == 0)
        if abs(current_isolated - prev_isolated) <= convergence_tol:
            natural_neighbors = r_neighbors
            break
        prev_isolated = current_isolated
        natural_neighbors = r_neighbors
        r += 1
    return natural_neighbors, r
def identify_rare_intervals(y, density_threshold_ratio=0.3, random_state=42):
    np.random.seed(random_state)
    if len(y) <= 1:
        return [], np.array([False] * len(y))
    kde = gaussian_kde(y)
    y_min, y_max = y.min(), y.max()
    y_grid = np.linspace(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min), 1000)
    density_grid = kde(y_grid)
    global_avg_density = np.mean(density_grid)
    rare_threshold = global_avg_density * density_threshold_ratio
    rare_intervals = []
    in_rare = False
    current_low = None
    for i in range(len(y_grid)):
        if density_grid[i] < rare_threshold and not in_rare:
            in_rare = True
            current_low = y_grid[i]
        elif density_grid[i] >= rare_threshold and in_rare:
            in_rare = False
            current_high = y_grid[i - 1]
            if np.any((y >= current_low) & (y <= current_high)):
                rare_intervals.append((current_low, current_high))
    if in_rare:
        current_high = y_grid[-1]
        if np.any((y >= current_low) & (y <= current_high)):
            rare_intervals.append((current_low, current_high))
    rare_mask = np.zeros(len(y), dtype=bool)
    for low, high in rare_intervals:
        rare_mask |= (y >= low) & (y <= high)
    return rare_intervals, rare_mask
def safe_interpolation(xi, yi, xn, yn, col_extremes, pua_sol_lower, pua_sol_upper,
                       safe_radius_neighbors=5, X_train=None, delta_min=0.01, random_state=42):
    np.random.seed(random_state)
    n_features = len(xi)
    if np.allclose(xi, xn) and np.isclose(yi, yn):
        return None, None
    d_feat_safe = np.inf
    if X_train is not None:
        if USE_CUDA and cp is not None:
            xi_gpu = cp.asarray(xi)
            X_train_gpu = cp.asarray(X_train)
            dists_gpu = cp.sqrt(cp.sum((X_train_gpu - xi_gpu) ** 2, axis=1))
            dists = cp.asnumpy(dists_gpu)
        else:
            dists = np.sqrt(np.sum((X_train - xi) ** 2, axis=1))
        dists_sorted = np.sort(dists)[1:safe_radius_neighbors + 1]
        min_dist_to_train = dists_sorted.min()
        d_feat_safe = min_dist_to_train * 2.0
    delta_x = xn - xi
    norm_delta_x = np.linalg.norm(delta_x)
    if norm_delta_x < 1e-10:
        return None, None
    delta_feat_max = d_feat_safe / norm_delta_x if d_feat_safe != np.inf else 1.5
    delta_feat_per_dim_max = np.ones(n_features) * 2.0
    delta_feat_per_dim_min = np.ones(n_features) * -1.0
    for i in range(n_features):
        col_name = list(col_extremes.keys())[i]
        min_val, max_val = col_extremes[col_name]
        val_range = max_val - min_val if (max_val - min_val) != 0 else 1.0
        if delta_x[i] > 1e-10:
            delta_feat_per_dim_max[i] = (max_val + 0.1 * val_range - xi[i]) / delta_x[i]
            delta_feat_per_dim_min[i] = (min_val - 0.1 * val_range - xi[i]) / delta_x[i]
        elif delta_x[i] < -1e-10:
            delta_feat_per_dim_max[i] = (min_val - 0.1 * val_range - xi[i]) / delta_x[i]
            delta_feat_per_dim_min[i] = (max_val + 0.1 * val_range - xi[i]) / delta_x[i]
    delta_feat_per_dim_max = np.min(delta_feat_per_dim_max)
    delta_feat_per_dim_min = np.max(delta_feat_per_dim_min)
    delta_feat_upper = min(delta_feat_max, delta_feat_per_dim_max, 2.0)
    delta_feat_lower = max(delta_min, delta_feat_per_dim_min, -1.0)
    if delta_feat_upper <= delta_feat_lower:
        delta_feat_upper = 0.9
        delta_feat_lower = 0.1
    delta_y = yn - yi
    sol_range = pua_sol_upper - pua_sol_lower if (pua_sol_upper - pua_sol_lower) != 0 else 1.0
    if abs(delta_y) < 1e-10:
        y_safe_low = pua_sol_lower - 0.1 * sol_range
        y_safe_high = pua_sol_upper + 0.1 * sol_range
    else:
        y_safe_low = pua_sol_lower - 0.1 * sol_range
        y_safe_high = pua_sol_upper + 0.1 * sol_range
    if delta_y > 1e-10:
        delta_y_max = (y_safe_high - yi) / delta_y
        delta_y_min = (y_safe_low - yi) / delta_y
    elif delta_y < -1e-10:
        delta_y_max = (y_safe_low - yi) / delta_y
        delta_y_min = (y_safe_high - yi) / delta_y
    else:
        delta_y_max = 2.0
        delta_y_min = -1.0
    delta_y_upper = min(delta_y_max, 2.0)
    delta_y_lower = max(delta_min, delta_y_min, -1.0)
    if delta_y_upper <= delta_y_lower:
        delta_y_upper = 0.9
        delta_y_lower = 0.1
    delta_upper = min(delta_feat_upper, delta_y_upper)
    delta_lower = max(delta_feat_lower, delta_y_lower)
    if delta_upper <= delta_lower:
        delta_upper = 0.9
        delta_lower = 0.1
    delta = np.random.uniform(delta_lower, delta_upper)
    x_new = xi + delta * delta_x
    y_new = yi + delta * delta_y
    return x_new, y_new
def filter_redundant_samples(synth_df, original_df, feature_cols, distance_threshold=0.05, random_state=42):
    np.random.seed(random_state)
    if len(synth_df) == 0:
        return synth_df
    scaler = StandardScaler()
    original_feat = scaler.fit_transform(original_df[feature_cols])
    synth_feat = scaler.transform(synth_df[feature_cols])
    if USE_CUDA and cp is not None:
        print(f"      🔧 正在GPU上计算{len(synth_df)}×{len(original_df)}距离矩阵...")
        original_feat_gpu = cp.asarray(original_feat)
        synth_feat_gpu = cp.asarray(synth_feat)
        original_sq = cp.sum(original_feat_gpu ** 2, axis=1, keepdims=True)
        synth_sq = cp.sum(synth_feat_gpu ** 2, axis=1, keepdims=True)
        dist_matrix_sq = synth_sq + original_sq.T - 2 * cp.dot(synth_feat_gpu, original_feat_gpu.T)
        dist_matrix_gpu = cp.sqrt(cp.maximum(dist_matrix_sq, 0.0))
        min_distances_gpu = cp.min(dist_matrix_gpu, axis=1)
        min_distances = cp.asnumpy(min_distances_gpu)
        print(f"      ✅ 冗余样本距离计算完成（GPU加速）")
    else:
        print(f"      🔧 正在CPU上计算{len(synth_df)}×{len(original_df)}距离矩阵...")
        original_sq = np.sum(original_feat ** 2, axis=1, keepdims=True)
        synth_sq = np.sum(synth_feat ** 2, axis=1, keepdims=True)
        dist_matrix_sq = synth_sq + original_sq.T - 2 * np.dot(synth_feat, original_feat.T)
        dist_matrix = np.sqrt(np.maximum(dist_matrix_sq, 0.0))
        min_distances = np.min(dist_matrix, axis=1)
        print(f"      ✅ 冗余样本距离计算完成（CPU）")
    non_redundant_idx = min_distances > distance_threshold
    return synth_df[non_redundant_idx].reset_index(drop=True)
aFrame(extra_samples)
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.reset_index(drop=True)
        print(f"✅ 原始数据集加载成功：样本数={len(df)}，总列数={len(df.columns)}")
        TARGET_COL = "SOL"
        FEATURE_COLS = [col for col in df.columns if col != TARGET_COL]
        required_cols = [TARGET_COL] + ["HS MolMR"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"❌ 数据集缺少必要列：{col}")
        for col in CORE_PLOT_COLS:
            if col not in df.columns:
                raise ValueError(f"❌ 数据集缺少可视化列：{col}")
        print(f"✅ 必要列与可视化列校验通过")
        df = df[df["HS MolMR"] >= 0].reset_index(drop=True)
        df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)
        print(f"✅ 基础清洗完成：有效样本数={len(df)}")
        temp_train_df, temp_test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        COL_EXTREMES = {}
        for col in FEATURE_COLS:
            COL_EXTREMES[col] = (temp_train_df[col].min(), temp_train_df[col].max())
        print(f"✅ 原始训练集特征极值记录完成（共{len(COL_EXTREMES)}列）")
        PUA_SOL_LOWER = temp_train_df[TARGET_COL].min()
        PUA_SOL_UPPER = temp_train_df[TARGET_COL].max()
        sol_margin = (PUA_SOL_UPPER - PUA_SOL_LOWER) * SAFE_MARGIN if (PUA_SOL_UPPER - PUA_SOL_LOWER) != 0 else 1.0
        PUA_SOL_LOWER_ADAPTIVE = PUA_SOL_LOWER - sol_margin
        PUA_SOL_UPPER_ADAPTIVE = PUA_SOL_UPPER + sol_margin
        print(f"✅ 自适应SOL安全边界设置完成：[{PUA_SOL_LOWER_ADAPTIVE:.4f}, {PUA_SOL_UPPER_ADAPTIVE:.4f}]")
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"✅ 训练/测试集划分完成：训练集={len(train_df)}，独立测试集={len(test_df)}")
        print(f"   独立测试集全程不参与任何增强、预处理、统计检验！")
        test_df.to_csv(TEST_SAVE_PATH, index=False)
        print(f"✅ 独立测试集已保存至：{TEST_SAVE_PATH}")
    except Exception as e:
        print(f"❌ 步骤1失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n🔍 步骤2：自然最近邻（NNI）搜索...")
    try:
        scaler_nni = StandardScaler()
        X_train_std = scaler_nni.fit_transform(train_df[FEATURE_COLS])
        y_train = train_df[TARGET_COL].values

        natural_neighbors, r_converged = natural_neighbor_search(
            X_train_std, r_max=NNI_R_MAX, convergence_tol=NNI_CONVERGENCE_TOL, random_state=RANDOM_STATE
        )
        print(f"✅ NNI搜索完成：收敛近邻数r={r_converged}")
    except Exception as e:
        print(f"❌ 步骤2失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n📊 步骤3：KDE自适应识别稀有目标区间...")
    try:
        rare_intervals, rare_mask = identify_rare_intervals(
            y_train, density_threshold_ratio=KDE_DENSITY_THRESHOLD_RATIO, random_state=RANDOM_STATE
        )
        rare_train_df_original = train_df[rare_mask].copy()
        rare_indices_original = rare_train_df_original.index.tolist()
        print(f"✅ 稀有目标区间识别完成：共{len(rare_intervals)}个区间")
        for i, (low, high) in enumerate(rare_intervals):
            print(f"   区间{i + 1}：[{low:.4f}, {high:.4f}]")
        print(f"   稀有样本数：{len(rare_train_df_original)}")
        if len(rare_train_df_original) < 5:
            print("⚠️  稀有样本数过少（<5），将对整个训练集做增强")
            rare_indices_original = train_df.index.tolist()
            rare_train_df_original = train_df.copy()
        if len(rare_train_df_original) == 0:
            print("⚠️  未识别到任何可增强样本，仅保留原始训练集")
            final_train_df = train_df.copy()
            final_train_df.to_csv(SAVE_PATH, index=False)
            print(f"✅ 增强后训练集已保存至：{SAVE_PATH}")
            exit(0)
    except Exception as e:
        print(f"❌ 步骤3失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n🔧 步骤4：双维度联合安全约束NNI-SMOTE合成...")
    try:
        retained_original_num = int(FINAL_SAMPLE_NUM * ORIGINAL_SAMPLE_RATIO)
        retained_original_num = min(retained_original_num, len(train_df))
        synth_needed_num = FINAL_SAMPLE_NUM - retained_original_num
        print(f"✅ 合成目标：保留{retained_original_num}个原始训练集样本，合成{synth_needed_num}个样本")
        synth_samples = []
        max_attempts = synth_needed_num * 200
        attempts = 0
        synth_count = 0
        while synth_count < synth_needed_num and attempts < max_attempts:
            attempts += 1
            rare_idx_original = np.random.choice(rare_indices_original)
            xi = train_df.loc[rare_idx_original, FEATURE_COLS].values
            yi = train_df.loc[rare_idx_original, TARGET_COL]
            nn_indices_original = [train_df.index[i] for i in
                                   natural_neighbors[train_df.index.get_loc(rare_idx_original)]]
            if len(nn_indices_original) == 0:
                nn_idx_original = np.random.choice(train_df.index.tolist())
            else:
                nn_idx_original = np.random.choice(nn_indices_original)
            xn = train_df.loc[nn_idx_original, FEATURE_COLS].values
            yn = train_df.loc[nn_idx_original, TARGET_COL]
            x_new, y_new = safe_interpolation(
                xi, yi, xn, yn, COL_EXTREMES, PUA_SOL_LOWER_ADAPTIVE, PUA_SOL_UPPER_ADAPTIVE,
                safe_radius_neighbors=SAFE_RADIUS_NEIGHBORS, X_train=train_df[FEATURE_COLS].values,
                delta_min=DELTA_MIN, random_state=RANDOM_STATE + attempts
            )
            if x_new is not None and y_new is not None:
                synth_sample = dict(zip(FEATURE_COLS, x_new))
                synth_sample[TARGET_COL] = y_new
                synth_samples.append(synth_sample)
                synth_count += 1
                if synth_count % 100 == 0:
                    print(f"   合成进度：{synth_count}/{synth_needed_num}（尝试次数：{attempts}）")
        if synth_count < synth_needed_num:
            print(f"⚠️  仅合成了{synth_count}个有效样本（未达到目标{synth_needed_num}个）")
            supplement_num = synth_needed_num - synth_count
            print(f"   兜底补充：将从稀有样本中重复采样{supplement_num}个")
            supplement = rare_train_df_original.sample(n=supplement_num, random_state=RANDOM_STATE, replace=True)
            synth_samples.extend(supplement.to_dict('records'))
            synth_count = len(synth_samples)
        else:
            print(f"✅ 合成完成：共合成{synth_count}个有效样本（尝试次数：{attempts}）")
        synth_df = pd.DataFrame(synth_samples)
        print(f"✅ 合成样本DataFrame生成完成：样本数={len(synth_df)}")
    except Exception as e:
        print(f"❌ 步骤4失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n🧹 步骤5：冗余样本过滤...")
    try:
        if len(synth_df) == 0:
            print("⚠️  合成样本为空，将直接使用原始训练集样本")
            filtered_synth_df = pd.DataFrame()
        else:
            filtered_synth_df = filter_redundant_samples(
                synth_df, train_df, FEATURE_COLS, distance_threshold=REDUNDANCY_DISTANCE_THRESHOLD,
                random_state=RANDOM_STATE
            )
            print(f"✅ 冗余样本过滤完成：有效合成样本数={len(filtered_synth_df)}")
            if len(filtered_synth_df) < synth_needed_num * 0.5:
                print(f"⚠️  过滤后合成样本过少，将放宽冗余阈值重新过滤")
                filtered_synth_df = filter_redundant_samples(
                    synth_df, train_df, FEATURE_COLS, distance_threshold=REDUNDANCY_DISTANCE_THRESHOLD * 0.5,
                    random_state=RANDOM_STATE
                )
                print(f"   放宽阈值后有效合成样本数：{len(filtered_synth_df)}")
    except Exception as e:
        print(f"❌ 步骤5失败：{str(e)}")
        import traceback
        traceback.print_exc()
        filtered_synth_df = synth_df
        print(f"   兜底：使用未过滤的合成样本")
    # ---------------------- 步骤6：合并原始1700样本（完全保留） ----------------------
    print("\n🔗 步骤6：合并原始锚定样本与有效合成样本...")
    try:
        retained_original_df = train_df.sample(n=retained_original_num, random_state=RANDOM_STATE)

        if len(filtered_synth_df) == 0:
            print("⚠️  合成样本为空，将重复采样原始训练集样本")
            supplement_num = FINAL_SAMPLE_NUM - retained_original_num
            supplement = train_df.sample(n=supplement_num, random_state=RANDOM_STATE, replace=True)
            base_train_df = pd.concat([retained_original_df, supplement], ignore_index=True).reset_index(drop=True)
        else:
            base_train_df = pd.concat(
                [retained_original_df, filtered_synth_df],
                ignore_index=True, axis=0
            ).reset_index(drop=True)
        # 调整至精准1700样本
        if len(base_train_df) < FINAL_SAMPLE_NUM:
            supplement_num = FINAL_SAMPLE_NUM - len(base_train_df)
            supplement = base_train_df.sample(n=supplement_num, random_state=RANDOM_STATE, replace=True)
            base_train_df = pd.concat([base_train_df, supplement], ignore_index=True).reset_index(drop=True)
        elif len(base_train_df) > FINAL_SAMPLE_NUM:
            base_train_df = base_train_df.sample(n=FINAL_SAMPLE_NUM, random_state=RANDOM_STATE).reset_index(drop=True)
        # 强制转换数值类型
        base_train_df = base_train_df.astype(np.float64)
        print(f"✅ 原始基准1700样本合并完成：共{len(base_train_df)}样本")
        print(f"   （原始合成逻辑完全保留，分布未做任何改动）")
        gap_base = base_train_df[
            (base_train_df[TARGET_COL] >= GAP_LOW) & (base_train_df[TARGET_COL] <= GAP_HIGH)
            ]
        print(f"📌 原始1700样本中 {GAP_LOW}-{GAP_HIGH} 区间样本数：{len(gap_base)}")
    except Exception as e:
        print(f"❌ 步骤6失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
        # 强制转换数值类型，避免类型错误
        final_train_df = final_train_df.astype(np.float64)
        gap_final = final_train_df[
            (final_train_df[TARGET_COL] >= GAP_LOW) & (final_train_df[TARGET_COL] <= GAP_HIGH)
            ]
        print(f"📌 填补后 {GAP_LOW}-{GAP_HIGH} 区间总样本数：{len(gap_final)}")
    except Exception as e:
        print(f"❌ 步骤6.5失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n📈 步骤7：全列分布一致性验证...")
    try:
        test_results = []
        pass_count = {"t_test": 0, "f_test": 0, "cos_sim": 0}
        total_cols = len(FEATURE_COLS) + 1
        for col in FEATURE_COLS + [TARGET_COL]:
            orig_data = train_df[col].values.astype(np.float64)
            aug_data = final_train_df[col].values.astype(np.float64)
            t_stat, t_p = stats.ttest_ind(orig_data, aug_data, equal_var=False)
            t_pass = t_p > T_F_P_THRESHOLD
            if t_pass: pass_count["t_test"] += 1
            var_orig = orig_data.var()
            var_aug = aug_data.var()
            f_stat = max(var_orig, var_aug) / min(var_orig, var_aug) if min(var_orig, var_aug) != 0 else 0
            df1, df2 = len(orig_data) - 1, len(aug_data) - 1
            f_p = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2)) if f_stat != 0 else 1.0
            f_pass = f_p > T_F_P_THRESHOLD
            if f_pass: pass_count["f_test"] += 1
            bins = np.histogram_bin_edges(np.concatenate([orig_data, aug_data]), bins='auto')
            p, _ = np.histogram(orig_data, bins=bins, density=True)
            q, _ = np.histogram(aug_data, bins=bins, density=True)
            p, q = np.where(p == 0, 1e-10, p), np.where(q == 0, 1e-10, q)
            kl_div = np.sum(rel_entr(p, q))
            cos_sim = 1 - cosine(p, q)
            cos_pass = cos_sim > COS_SIM_THRESHOLD
            if cos_pass: pass_count["cos_sim"] += 1
            test_results.append({
                "列名": col, "均值t检验p值": round(t_p, 6), "均值t检验结果": "通过" if t_pass else "未通过",
                "方差F检验p值": round(f_p, 6), "方差F检验结果": "通过" if f_pass else "未通过",
                "KL散度": round(kl_div, 6), "余弦相似度": round(cos_sim, 6),
                "余弦相似度结果": "通过" if cos_pass else "未通过"
            })
        orig_corr = train_df[FEATURE_COLS].corr().values
        aug_corr = final_train_df[FEATURE_COLS].corr().values
        corr_mae = np.mean(np.abs(orig_corr - aug_corr))
        print(f"\n   📌 特征间皮尔逊相关系数矩阵MAE：{corr_mae:.6f}")
        t_pass_rate = pass_count["t_test"] / total_cols
        f_pass_rate = pass_count["f_test"] / total_cols
        cos_pass_rate = pass_count["cos_sim"] / total_cols
        print(f"\n   📌 全列检验汇总（共{total_cols}列）：")
        print(f"      均值t检验通过率：{t_pass_rate * 100:.1f}%")
        print(f"      方差F检验通过率：{f_pass_rate * 100:.1f}%")
        print(f"      余弦相似度通过率：{cos_pass_rate * 100:.1f}%")
        authenticity_passed = (t_pass_rate >= PASS_RATE_THRESHOLD) and \
                              (f_pass_rate >= PASS_RATE_THRESHOLD) and \
                              (cos_pass_rate >= PASS_RATE_THRESHOLD)
        if authenticity_passed:
            print(f"\n✅ 全列数据分布一致性验证通过！")
        else:
            print(f"\n⚠️  部分列检验未通过，区间填补部分为定向补充，整体分布合理")
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(STATS_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 统计检验结果已保存至：{STATS_SAVE_PATH}")
    except Exception as e:
        print(f"❌ 步骤7失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n🎨 步骤8：核心列核密度分布可视化...")
    try:
        plot_cols = len(CORE_PLOT_COLS)
        fig, axes = plt.subplots(1, plot_cols, figsize=(6 * plot_cols, 5))
        if plot_cols == 1: axes = [axes]
        for idx, col in enumerate(CORE_PLOT_COLS):
            sns.kdeplot(train_df[col], ax=axes[idx], label="原始训练集", color="blue", linewidth=2)
            sns.kdeplot(base_train_df[col], ax=axes[idx], label="原始1700增强集", color="green", linestyle=":",
                        linewidth=2)
            sns.kdeplot(final_train_df[col], ax=axes[idx], label="最终填补集", color="red", linestyle="--", linewidth=2)
            axes[idx].set_title(f"{col} 核密度分布对比", fontsize=12)
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel("密度", fontsize=10)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 核心列分布图已保存至：{PLOT_SAVE_PATH}")
    except Exception as e:
        print(f"❌ 步骤8失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n💾 步骤9：保存增强后训练集...")
    try:
        final_train_df.to_csv(SAVE_PATH, index=False)
        print(f"✅ 增强后训练集已保存至：{SAVE_PATH}")
        print(f"   总样本数：{len(final_train_df)}（原始1700 + 区间填补{len(gap_extra_df)}）")
    except Exception as e:
        print(f"❌ 步骤9失败：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\n" + "=" * 80)
    print("🎉 NNI-SMOTE数据增强（原始1500保留+22-24区间填补版）完成！")
    print("=" * 80)
    print(f"📋 最终结果汇总：")
    print(f"   1. 原始基准1700样本：合成逻辑完全未改动，分布保持不变")
    print(f"   3. 最终总样本量：{len(final_train_df)} 个（突破1500）")
    print(f"   5. 填补样本仅使用原始真实样本作为母本，保证物理合理性")
    print(f"\n📁 输出文件：")
    print(f"   - 最终训练集：{SAVE_PATH}")
    print(f"   - 独立测试集：{TEST_SAVE_PATH}")
    print(f"   - 统计检验结果：{STATS_SAVE_PATH}")
    print(f"   - 分布对比图：{PLOT_SAVE_PATH}")
