import pandas as pd
import numpy as np
import warnings
import os
import random

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# ===== 设置全局随机种子，保证结果可复现 =====
np.random.seed(27)
random.seed(27)

warnings.filterwarnings("ignore")

# =========================================================
# 1. 参数
# =========================================================
N_CANDIDATES = 1000
TEST_RATIO_MIN = 0.15
TEST_RATIO_MAX = 0.25
MAX_OVERLAP = 0.8
N_FINAL_SPLITS = 10
VAL_RATIO_MIN = 0.15         # 验证集占 train_val 的比例下限
VAL_RATIO_MAX = 0.25         # 验证集占 train_val 的比例上限
TOP_N = 26                   # 固定使用前 26 个特征

# SVM 网格搜索参数（可根据需要调整）
PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

# =========================================================
# 2. 文件路径
# =========================================================
file_path = "../data/all_binding_marker_matrix_with_subtype.csv"
output_dir = "Nested_SVM_Top26_GridSearch"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_csv(file_path)

df["label"] = df["label"].map({"α2-6 receptor": 1, "α2-3 receptor": 0})

X = df.drop(columns=["Strain", "Subtype", "label"], errors="ignore")
y = df["label"]
groups = df["Subtype"]

print("="*100)
print("Feature Shape:", X.shape)
print("Unique Subtypes:", groups.nunique())
print("="*100)

# =========================================================
# 4. 评估函数
# =========================================================
def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob)
    }

# =========================================================
# 5. 生成候选测试集（基于Subtype，比例在0.15~0.25）
# =========================================================
candidate_splits = []

unique_subtypes = np.unique(groups)
subtype_to_indices = {s: np.where(groups == s)[0] for s in unique_subtypes}
total_samples = len(X)

for _ in range(N_CANDIDATES * 5):
    n_test_subtypes = np.random.randint(1, len(unique_subtypes))
    test_subtypes = np.random.choice(unique_subtypes, size=n_test_subtypes, replace=False)
    test_idx = np.concatenate([subtype_to_indices[s] for s in test_subtypes])
    test_ratio = len(test_idx) / total_samples
    if TEST_RATIO_MIN <= test_ratio <= TEST_RATIO_MAX:
        train_idx = np.setdiff1d(np.arange(total_samples), test_idx)
        if len(np.unique(y.iloc[train_idx])) < 2 or len(np.unique(y.iloc[test_idx])) < 2:
            continue
        if len(np.unique(groups.iloc[train_idx])) < 2:
            continue
        candidate_splits.append(test_idx)
        if len(candidate_splits) >= N_CANDIDATES:
            break

print(f"生成候选测试集数量: {len(candidate_splits)}")

# =========================================================
# 6. 筛选低重叠测试集（Jaccard重叠率 ≤ 0.8）
# =========================================================
final_splits = []

for idx in candidate_splits:
    test_subtypes = set(groups.iloc[idx])
    overlap_flag = False
    for fidx in final_splits:
        existing_subtypes = set(groups.iloc[fidx])
        overlap = len(test_subtypes & existing_subtypes) / len(test_subtypes | existing_subtypes)
        if overlap > MAX_OVERLAP:
            overlap_flag = True
            break
    if not overlap_flag:
        final_splits.append(idx)
    if len(final_splits) >= N_FINAL_SPLITS:
        break

print(f"筛选后独立测试集数量: {len(final_splits)}")

# =========================================================
# 7. 严格划分训练集和验证集（按Subtype，验证集占 train_val 15%~25%）
# =========================================================
def split_train_val_by_subtype(X, y, groups, min_ratio, max_ratio,
                               min_classes=2, random_state=None, max_trials=500):
    """
    尝试多次随机排列 Subtype，找到一个划分，使得：
      - 验证集样本数 / train_val 总样本数 在 [min_ratio, max_ratio] 内
      - 训练集和验证集都至少包含 min_classes 个类别
    返回 (train_idx, val_idx)，若找不到则返回 (None, None)
    """
    unique_groups = groups.unique()
    rng = np.random.RandomState(random_state)
    total = len(X)

    for trial in range(max_trials):
        shuffled = rng.permutation(unique_groups)
        cumsum = 0
        val_groups = []
        for g in shuffled:
            idx = np.where(groups == g)[0]
            new_cumsum = cumsum + len(idx)
            new_ratio = new_cumsum / total
            if new_ratio > max_ratio:
                continue
            cumsum = new_cumsum
            val_groups.append(g)
            current_ratio = cumsum / total
            if current_ratio >= min_ratio:
                val_idx = np.concatenate([np.where(groups == g)[0] for g in val_groups])
                train_idx = np.setdiff1d(np.arange(total), val_idx)
                if (len(np.unique(y.iloc[train_idx])) >= min_classes and
                    len(np.unique(y.iloc[val_idx])) >= min_classes):
                    return train_idx, val_idx
    return None, None

# =========================================================
# 8. 外层实验：嵌套式特征选择 + 网格搜索 + 最终测试
# =========================================================
all_results = []
skipped_runs = 0

for run, test_idx in enumerate(final_splits):
    print("\n" + "="*100)
    print(f"Random Split Run {run+1}")
    print("="*100)

    # 外层：train_val / test
    train_val_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("Train-Val Subtypes:", len(groups_train_val.unique()))
    print("Test Subtypes:", len(groups.iloc[test_idx].unique()))
    print("Test ratio (占整体):", len(test_idx)/len(X))
    print("Train-Val size:", len(X_train_val))
    print("Test size:", len(X_test))

    # ---------- 内层：严格划分 train / val ----------
    train_local_idx, val_local_idx = split_train_val_by_subtype(
        X_train_val, y_train_val, groups_train_val,
        min_ratio=VAL_RATIO_MIN,
        max_ratio=VAL_RATIO_MAX,
        random_state=run+200,
        max_trials=500
    )

    if train_local_idx is None:
        print(f"⚠️ 无法找到满足验证集比例 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}] 的划分，跳过本轮。")
        skipped_runs += 1
        continue

    X_train = X_train_val.iloc[train_local_idx]
    y_train = y_train_val.iloc[train_local_idx]
    X_val = X_train_val.iloc[val_local_idx]
    y_val = y_train_val.iloc[val_local_idx]

    actual_val_ratio = len(X_val) / len(X_train_val)
    print(f"内层划分：Train size={len(X_train)}, Val size={len(X_val)}, Val ratio={actual_val_ratio:.4f}")

    # ---------- 阶段一：在训练集上选 26 特征 + 网格搜索 SVM 参数 ----------
    # 1) 在训练集上计算 RF 重要性，取前 TOP_N 个特征
    rf_importance_inner = RandomForestClassifier(random_state=123, n_jobs=-1)
    rf_importance_inner.fit(X_train, y_train)
    importance_inner = rf_importance_inner.feature_importances_
    importance_df_inner = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance_inner
    }).sort_values(by="Importance", ascending=False)
    ranked_features_inner = importance_df_inner["Feature"].tolist()
    n_select_inner = min(TOP_N, len(ranked_features_inner))
    selected_features_inner = ranked_features_inner[:n_select_inner]
    print(f"内层选择特征数: {n_select_inner}")

    # 2) 标准化（在训练集上拟合 scaler）
    scaler_inner = StandardScaler()
    X_train_sel = X_train[selected_features_inner]
    X_val_sel = X_val[selected_features_inner]
    X_train_scaled = scaler_inner.fit_transform(X_train_sel)
    X_val_scaled = scaler_inner.transform(X_val_sel)

    # 3) 网格搜索 SVM，以验证集 AUROC 最大为准则
    best_auroc = -np.inf
    best_params = None
    for params in ParameterGrid(PARAM_GRID):
        model = SVC(probability=True, random_state=123, **params)
        model.fit(X_train_scaled, y_train)
        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
        auroc = roc_auc_score(y_val, y_val_prob)
        if auroc > best_auroc:
            best_auroc = auroc
            best_params = params

    print(f"内层最佳验证 AUROC: {best_auroc:.4f}")
    print(f"最佳参数: {best_params}")

    # ---------- 阶段二：在 train_val 上重新选 26 特征，用最佳参数训练，测试 ----------
    # 1) 在 train_val 上计算 RF 重要性，取前 TOP_N 个特征
    rf_importance_outer = RandomForestClassifier(random_state=123, n_jobs=-1)
    rf_importance_outer.fit(X_train_val, y_train_val)
    importance_outer = rf_importance_outer.feature_importances_
    importance_df_outer = pd.DataFrame({
        "Feature": X_train_val.columns,
        "Importance": importance_outer
    }).sort_values(by="Importance", ascending=False)
    ranked_features_outer = importance_df_outer["Feature"].tolist()
    n_select_outer = min(TOP_N, len(ranked_features_outer))
    selected_features_outer = ranked_features_outer[:n_select_outer]
    print(f"外层重新选择特征数: {n_select_outer}")

    # 2) 标准化（在 train_val 上拟合 scaler）
    scaler_outer = StandardScaler()
    X_train_val_sel = X_train_val[selected_features_outer]
    X_test_sel = X_test[selected_features_outer]
    X_train_val_scaled = scaler_outer.fit_transform(X_train_val_sel)
    X_test_scaled = scaler_outer.transform(X_test_sel)

    # 3) 用最佳参数训练最终 SVM
    final_model = SVC(probability=True, random_state=123, **best_params)
    final_model.fit(X_train_val_scaled, y_train_val)

    # 4) 测试集评估
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate(y_test, y_test_prob)

    print("\nTest Result:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    result = {
        "Run": run + 1,
        "Best_Val_AUROC": best_auroc,
        "Best_Params": str(best_params),
        "TopN_inner": n_select_inner,
        "TopN_outer": n_select_outer,
        "TrainVal_Size": len(X_train_val),
        "Test_Size": len(X_test),
        **{f"Test_{k}": v for k, v in test_metrics.items()}
    }
    all_results.append(result)

# =========================================================
# 9. 汇总与保存
# =========================================================
if len(all_results) == 0:
    print("\n" + "="*100)
    print("❌ 所有轮次均无法找到满足验证集比例要求的划分，无结果保存。")
    print("="*100)
else:
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*100)
    print("All Results")
    print("="*100)
    print(results_df)

    # 对 Test_* 指标求均值和标准差
    test_metric_cols = [c for c in results_df.columns if c.startswith("Test_")]
    summary_mean = results_df[test_metric_cols].mean()
    summary_std = results_df[test_metric_cols].std()

    print("\nMean Test Results (across runs)")
    print(summary_mean)
    print("\nStd Test Results (across runs)")
    print(summary_std)

    results_df.to_csv(f"{output_dir}/All_Results_Nested_SVM_Top{TOP_N}.csv", index=False)
    summary_mean.to_frame(name="Mean").to_csv(f"{output_dir}/Summary_Mean_Nested_SVM_Top{TOP_N}.csv")
    summary_std.to_frame(name="Std").to_csv(f"{output_dir}/Summary_STD_Nested_SVM_Top{TOP_N}.csv")

    best_run = results_df.loc[results_df["Test_AUROC"].idxmax()]
    print("\n" + "="*100)
    print("Best Run (by Test AUROC)")
    print("="*100)
    print(best_run)

print("\n" + "="*100)
print(f"Nested SVM with TopN={TOP_N} Finished. 成功运行 {len(all_results)} 轮，跳过 {skipped_runs} 轮。")
print("="*100)