import pandas as pd
import numpy as np
import warnings
import random
import os

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(123)
random.seed(123)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, roc_curve, auc
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. 参数设置
# =========================================================
N_CANDIDATES = 5000          # 候选尝试次数
TEST_RATIO_MIN = 0.10
TEST_RATIO_MAX = 0.30
VAL_RATIO_MIN = 0.10
VAL_RATIO_MAX = 0.30
MAX_OVERLAP = 0.8
N_FINAL_SPLITS = 10
MIN_TRAIN_RATIO = 0.50       # 训练集最小比例

# 输出目录
OUTPUT_DIR = "Threshold_MarkerCount_ClusterSplit_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 读取数据
# =========================================================
file_path = "../data/markers_with_cluster.csv"
df = pd.read_csv(file_path)

print("=" * 100)
print("Columns:")
print(df.columns.tolist())
print("=" * 100)

# 标签映射
label_map = {"Virulent": 1, "Avirulent": 0}
df["label"] = df["label"].map(label_map)

X = df.drop(columns=["Strain", "Subtype", "label", "id", "subtype", "Cluster", "cluster"], errors="ignore")
y = df["label"]
groups = df["cluster"]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Clusters:", groups.nunique())

# =========================================================
# 3. 评估函数（二分类，含阈值）
# =========================================================
def evaluate_binary(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score)
    }

# =========================================================
# 4. 候选划分生成（基于 Cluster，三者互斥）
# =========================================================
def generate_candidate_splits(X, y, groups, n_candidates,
                              test_ratio_min, test_ratio_max,
                              val_ratio_min, val_ratio_max,
                              min_train_ratio):
    unique_groups = groups.unique()
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    total = len(X)
    candidates = []

    for _ in range(n_candidates):
        # 随机选择测试 Cluster
        n_test_groups = np.random.randint(1, len(unique_groups))
        test_clusters = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        test_idx = np.concatenate([group_to_indices[g] for g in test_clusters])
        test_ratio = len(test_idx) / total

        if not (test_ratio_min <= test_ratio <= test_ratio_max):
            continue

        # 剩余 Cluster
        remaining_clusters = [g for g in unique_groups if g not in test_clusters]
        if len(remaining_clusters) < 2:
            continue

        # 随机选择验证 Cluster（从剩余中选）
        n_val_groups = np.random.randint(1, len(remaining_clusters))
        val_clusters = np.random.choice(remaining_clusters, size=n_val_groups, replace=False)
        val_idx = np.concatenate([group_to_indices[g] for g in val_clusters])
        val_ratio = len(val_idx) / total

        if not (val_ratio_min <= val_ratio <= val_ratio_max):
            continue

        # 训练集为剩余的 Cluster
        train_clusters = [g for g in remaining_clusters if g not in val_clusters]
        if len(train_clusters) == 0:
            continue
        train_idx = np.concatenate([group_to_indices[g] for g in train_clusters])
        train_ratio = len(train_idx) / total

        if train_ratio <= min_train_ratio:
            continue

        # 检查三个集合的标签是否都包含两类
        if (len(np.unique(y.iloc[train_idx])) < 2 or
            len(np.unique(y.iloc[val_idx])) < 2 or
            len(np.unique(y.iloc[test_idx])) < 2):
            continue

        candidates.append((train_idx, val_idx, test_idx, test_clusters))

    return candidates

def select_final_splits(candidates, n_splits, max_overlap):
    final = []
    for train_idx, val_idx, test_idx, test_clusters in candidates:
        test_set = set(test_clusters)
        overlap = False
        for existing_train, existing_val, existing_test, existing_clusters in final:
            existing_set = set(existing_clusters)
            jaccard = len(test_set & existing_set) / len(test_set | existing_set)
            if jaccard > max_overlap:
                overlap = True
                break
        if not overlap:
            final.append((train_idx, val_idx, test_idx, test_clusters))
            if len(final) >= n_splits:
                break
    return final

print("生成候选划分...")
candidates = generate_candidate_splits(X, y, groups, N_CANDIDATES,
                                       TEST_RATIO_MIN, TEST_RATIO_MAX,
                                       VAL_RATIO_MIN, VAL_RATIO_MAX,
                                       MIN_TRAIN_RATIO)
print(f"候选数量: {len(candidates)}")

final_splits = select_final_splits(candidates, N_FINAL_SPLITS, MAX_OVERLAP)
print(f"最终选中 {len(final_splits)} 个独立划分")

if len(final_splits) == 0:
    raise RuntimeError("未能选出任何满足条件的划分。")

# =========================================================
# 5. 主循环：对每个划分，用标志物数目作为得分，找最优阈值
# =========================================================
all_test_results = []

for run, (train_idx, val_idx, test_idx, test_clusters) in enumerate(final_splits):
    print("\n" + "=" * 100)
    print(f"Run {run+1}")
    print("=" * 100)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Train size: {len(X_train)} (Clusters: {len(np.unique(groups.iloc[train_idx]))})")
    print(f"Val size:   {len(X_val)} (Clusters: {len(np.unique(groups.iloc[val_idx]))})")
    print(f"Test size:  {len(X_test)} (Clusters: {len(np.unique(groups.iloc[test_idx]))})")
    print(f"Train ratio: {len(X_train)/len(X):.2%}")
    print(f"Val ratio:   {len(X_val)/len(X):.2%}")
    print(f"Test ratio:  {len(X_test)/len(X):.2%}")

    # =====================================================
    # 用标志物数目（Marker_Count）作为预测得分
    # =====================================================
    scores_train = X_train.sum(axis=1).values
    scores_val = X_val.sum(axis=1).values
    scores_test = X_test.sum(axis=1).values

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    # ---- 在训练+验证合并集上找最优阈值 ----
    scores_train_val = np.concatenate([scores_train, scores_val])
    y_train_val = np.concatenate([y_train_arr, y_val_arr])

    fpr, tpr, thresholds = roc_curve(y_train_val, scores_train_val)
    auroc_train_val = auc(fpr, tpr)

    # Youden's J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # 处理 inf 阈值（理论上不会发生，但防御性处理）
    if not np.isfinite(best_threshold):
        finite_idx = np.where(np.isfinite(thresholds))[0]
        if len(finite_idx) > 0:
            best_idx = finite_idx[np.argmax(j_scores[finite_idx])]
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = scores_train_val.mean()

    # 训练+验证集上的分类指标（用该阈值）
    train_val_metrics = evaluate_binary(y_train_val, scores_train_val, best_threshold)

    print(f"\n训练+验证集 AUROC: {auroc_train_val:.4f}")
    print(f"最优阈值: {best_threshold:.4f}")
    print(f"训练+验证集 ACC: {train_val_metrics['ACC']:.4f}, "
          f"F1: {train_val_metrics['F1']:.4f}, MCC: {train_val_metrics['MCC']:.4f}")

    # ---- 测试集评估：用同一阈值 ----
    test_metrics = evaluate_binary(y_test_arr, scores_test, best_threshold)

    print("\n测试集结果:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- 保存结果 ----
    test_record = {
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
        "Test_AUROC": test_metrics["AUROC"],
        "Test_AUPRC": test_metrics["AUPRC"]
    }
    all_test_results.append(test_record)

# =========================================================
# 6. 汇总测试结果
# =========================================================
if len(all_test_results) == 0:
    print("\n" + "=" * 100)
    print("❌ 没有成功运行的划分，无结果保存。")
    print("=" * 100)
else:
    results_df = pd.DataFrame(all_test_results)
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

    # 保存结果
    results_df.to_csv(f"{OUTPUT_DIR}/Threshold_MarkerCount_Test_Results.csv", index=False)
    summary_mean.to_csv(f"{OUTPUT_DIR}/Summary_Mean.csv")
    summary_std.to_csv(f"{OUTPUT_DIR}/Summary_STD.csv")

    print("\n结果已保存至目录:", OUTPUT_DIR)

print("=" * 100)