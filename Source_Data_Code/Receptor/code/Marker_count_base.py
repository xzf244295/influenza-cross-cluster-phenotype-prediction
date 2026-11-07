import pandas as pd
import numpy as np
import warnings
import random
import os

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(27)
random.seed(27)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, roc_curve, auc
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. 参数设置
# =========================================================
N_CANDIDATES = 1000
TEST_RATIO_MIN = 0.15
TEST_RATIO_MAX = 0.25
MAX_OVERLAP = 0.8
N_FINAL_SPLITS = 10
VAL_RATIO_MIN = 0.15   # 验证集占 train_val 的比例下限
VAL_RATIO_MAX = 0.25   # 验证集占 train_val 的比例上限

OUTPUT_DIR = "Threshold_MarkerCount_Receptor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 读取数据
# =========================================================
file_path = "../data/all_binding_marker_matrix_with_subtype.csv"
df = pd.read_csv(file_path)

print("=" * 100)
print("Columns:")
print(df.columns.tolist())
print("=" * 100)

if "Subtype" not in df.columns:
    raise ValueError("数据中缺少 'Subtype' 列")

# 标签映射
label_map = {"α2-6 receptor": 1, "α2-3 receptor": 0}
df["label"] = df["label"].map(label_map)

# ---- 【特征为标志物的数目】 ----
X_raw = df.drop(columns=["Strain", "Subtype", "label"], errors="ignore")
X = X_raw.sum(axis=1).to_frame(name="Marker_Count").astype(float)

y = df["label"]
groups = df["Subtype"]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Subtypes:", groups.nunique())
print("标签分布:", y.value_counts().to_dict())

# =========================================================
# 3. 评估函数（二分类）
# =========================================================
def evaluate(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# =========================================================
# 4. 候选测试集生成与筛选（基于 Subtype，重叠率 ≤ 0.8）
# =========================================================
def generate_candidate_splits(X, y, groups, n_candidates, test_ratio_min, test_ratio_max, min_train_subtypes=2):
    unique_groups = groups.unique()
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    total = len(X)
    candidates = []
    for _ in range(n_candidates):
        n_test_groups = np.random.randint(1, len(unique_groups))
        test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        test_idx = np.concatenate([group_to_indices[g] for g in test_groups])
        test_ratio = len(test_idx) / total
        if test_ratio_min <= test_ratio <= test_ratio_max:
            train_idx = np.setdiff1d(np.arange(total), test_idx)
            if len(np.unique(y.iloc[train_idx])) < 2 or len(np.unique(y.iloc[test_idx])) < 2:
                continue
            if len(np.unique(groups.iloc[train_idx])) < min_train_subtypes:
                continue
            candidates.append((test_idx, test_groups))
    return candidates

def select_final_splits(candidates, n_splits, max_overlap, groups):
    final = []
    for test_idx, test_groups in candidates:
        test_set = set(test_groups)
        overlap = False
        for existing_idx in final:
            existing_set = set(groups.iloc[existing_idx])
            jaccard = len(test_set & existing_set) / len(test_set | existing_set)
            if jaccard > max_overlap:
                overlap = True
                break
        if not overlap:
            final.append(test_idx)
            if len(final) >= n_splits:
                break
    return final

print("\n生成候选测试集...")
candidates = generate_candidate_splits(X, y, groups, N_CANDIDATES, TEST_RATIO_MIN, TEST_RATIO_MAX)
print(f"候选数量: {len(candidates)}")
final_splits = select_final_splits(candidates, N_FINAL_SPLITS, MAX_OVERLAP, groups)
print(f"最终选中 {len(final_splits)} 个独立测试集")

if len(final_splits) == 0:
    raise RuntimeError("未能选出任何满足条件的测试集。")

# =========================================================
# 5. 训练/验证划分函数
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
# 6. 外层循环（对每个独立测试集）
# =========================================================
all_results = []
skipped_runs = 0

for run, test_idx in enumerate(final_splits):
    print("\n" + "=" * 100)
    print(f"Random Split Run {run+1}")
    print("=" * 100)

    train_val_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Train_Val size: {len(X_train_val)} (Subtypes: {groups_train_val.nunique()})")
    print(f"Test size: {len(X_test)} (Subtypes: {groups.iloc[test_idx].nunique()})")

    # 调用划分函数，强制验证集占 train_val 比例在 [VAL_RATIO_MIN, VAL_RATIO_MAX]
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
    print(f"Train size: {len(X_train)} (Subtypes: {groups_train_val.iloc[train_local_idx].nunique()})")
    print(f"Val size: {len(X_val)} (Subtypes: {groups_train_val.iloc[val_local_idx].nunique()})")
    print(f"实际验证集占 train_val 比例: {actual_val_ratio:.4f} (在要求范围内)")
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Val label distribution: {y_val.value_counts().to_dict()}")

    # =====================================================
    # 在训练+验证合并集上，寻找最优阈值
    # =====================================================
    X_train_val_combined = pd.concat([X_train, X_val], axis=0)
    y_train_val_combined = pd.concat([y_train, y_val], axis=0)

    scores_train_val = X_train_val_combined["Marker_Count"].values
    scores_test = X_test["Marker_Count"].values

    print("\n--- 在训练+验证集上寻找最优阈值 ---")

    # 训练+验证集上计算 ROC
    fpr, tpr, thresholds = roc_curve(y_train_val_combined, scores_train_val)
    auroc_train_val = auc(fpr, tpr)

    # 用 Youden's J = TPR - FPR 找最佳阈值
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # 若最优阈值是 inf（例如所有样本都超过），取次优
    if not np.isfinite(best_threshold):
        finite_idx = np.where(np.isfinite(thresholds))[0]
        if len(finite_idx) > 0:
            best_idx = finite_idx[np.argmax(j_scores[finite_idx])]
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = scores_train_val.mean()

    # 同时计算训练+验证集上的分类指标（用该阈值）
    y_pred_train_val = (scores_train_val >= best_threshold).astype(int)
    train_val_metrics = evaluate(y_train_val_combined, y_pred_train_val)

    print(f"  训练+验证集 AUROC: {auroc_train_val:.4f}")
    print(f"  最优阈值: {best_threshold:.4f}")
    print(f"  训练+验证集 ACC: {train_val_metrics['ACC']:.4f}, F1: {train_val_metrics['F1']:.4f}, MCC: {train_val_metrics['MCC']:.4f}")

    # =====================================================
    # 测试集评估：用该阈值二分类
    # =====================================================
    y_pred_test = (scores_test >= best_threshold).astype(int)
    test_metrics = evaluate(y_test, y_pred_test)

    # 测试集 AUROC / AUPRC 用连续分数（Marker_Count）计算
    test_auroc = roc_auc_score(y_test, scores_test)
    test_auprc = average_precision_score(y_test, scores_test)

    print("\nTest Result:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  AUROC: {test_auroc:.4f}")
    print(f"  AUPRC: {test_auprc:.4f}")

    # ---- 保存结果 ----
    result = {
        "Run": run + 1,
        "Best_Threshold": best_threshold,
        "TrainVal_AUROC": auroc_train_val,
        "TrainVal_ACC": train_val_metrics["ACC"],
        "TrainVal_Precision": train_val_metrics["Precision"],
        "TrainVal_Recall": train_val_metrics["Recall"],
        "TrainVal_F1": train_val_metrics["F1"],
        "TrainVal_MCC": train_val_metrics["MCC"],
        "Test_ACC": test_metrics["ACC"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
        "Test_MCC": test_metrics["MCC"],
        "Test_AUROC": test_auroc,
        "Test_AUPRC": test_auprc
    }
    all_results.append(result)

# =========================================================
# 7. 汇总结果
# =========================================================
if len(all_results) == 0:
    print("\n" + "=" * 100)
    print("❌ 所有轮次均无法找到满足验证集比例要求的划分，无结果保存。")
    print("=" * 100)
else:
    results_df = pd.DataFrame(all_results)
    summary_mean = results_df.mean(numeric_only=True)
    summary_std = results_df.std(numeric_only=True)

    print("\n" + "=" * 100)
    print("基于标志物数目的阈值分类结果（均值 ± 标准差）")
    print("=" * 100)

    print("\n阈值:")
    print(f"  Best_Threshold: {summary_mean['Best_Threshold']:.4f} ± {summary_std['Best_Threshold']:.4f}")

    print("\n训练+验证集指标:")
    for metric in ["TrainVal_AUROC", "TrainVal_ACC", "TrainVal_Precision",
                   "TrainVal_Recall", "TrainVal_F1", "TrainVal_MCC"]:
        print(f"  {metric}: {summary_mean[metric]:.4f} ± {summary_std[metric]:.4f}")

    print("\n测试集指标:")
    for metric in ["Test_ACC", "Test_Precision", "Test_Recall",
                   "Test_F1", "Test_MCC", "Test_AUROC", "Test_AUPRC"]:
        print(f"  {metric}: {summary_mean[metric]:.4f} ± {summary_std[metric]:.4f}")

    # 保存
    results_df.to_csv(f"{OUTPUT_DIR}/Threshold_MarkerCount_Receptor_Results.csv", index=False)
    summary_mean.to_csv(f"{OUTPUT_DIR}/Summary_Mean.csv")
    summary_std.to_csv(f"{OUTPUT_DIR}/Summary_STD.csv")

    print("\n结果已保存至目录:", OUTPUT_DIR)

print("\n" + "=" * 100)
print(f"实验结束。成功运行 {len(all_results)} 轮，跳过 {skipped_runs} 轮。")
print("=" * 100)