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
VAL_RATIO_MIN = 0.15
VAL_RATIO_MAX = 0.25

OUTPUT_DIR = "Threshold_MarkerCount_Host"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 读取数据
# =========================================================
file_path = "../data/host_markers_with_cluster.csv"
df = pd.read_csv(file_path)

print("=" * 100)
print("Columns:")
print(df.columns.tolist())
print("=" * 100)

if "Cluster" not in df.columns:
    raise ValueError("数据中缺少 'Cluster' 列")

# 标签映射（三分类）
label_map = {"Human": 0, "Avian": 1, "Swine": 2}
df["Host_class"] = df["Host_class"].map(label_map)

# ---- 特征为标志物的数目 ----
X_raw = df.drop(columns=["Strain", "Subtype", "Host_class", "Host", "Isolate_Id", "Cluster", "marker_count"], errors="ignore")
X = X_raw.sum(axis=1).to_frame(name="Marker_Count").astype(float)

y = df["Host_class"]
groups = df["Cluster"]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Subtypes:", groups.nunique())
print("标签分布:", y.value_counts().to_dict())

# =========================================================
# 3. 候选测试集生成与筛选
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
            if len(np.unique(y.iloc[train_idx])) < 3 or len(np.unique(y.iloc[test_idx])) < 3:
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
# 4. 严格模式验证集划分（基于 train_val）
# =========================================================
def split_train_val_by_subtype_strict(X, y, groups, min_ratio, max_ratio,
                                      min_classes=3, random_state=None, max_attempts=500):
    unique_groups = groups.unique()
    total = len(X)

    for attempt in range(max_attempts):
        rng = np.random.RandomState(random_state + attempt)
        shuffled = rng.permutation(unique_groups)

        cumsum = 0
        val_groups = []
        for g in shuffled:
            idx = np.where(groups == g)[0]
            new_ratio = (cumsum + len(idx)) / total
            if new_ratio > max_ratio:
                continue
            cumsum += len(idx)
            val_groups.append(g)
            if cumsum / total >= min_ratio:
                val_idx = np.concatenate([np.where(groups == g)[0] for g in val_groups])
                train_idx = np.setdiff1d(np.arange(total), val_idx)
                if (len(np.unique(y.iloc[train_idx])) >= min_classes and
                    len(np.unique(y.iloc[val_idx])) >= min_classes):
                    return train_idx, val_idx
    return None, None

# =========================================================
# 5. 外层循环
# =========================================================
class_names = {0: "Human", 1: "Avian", 2: "Swine"}

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

    # ---- 严格划分验证集 ----
    train_local_idx, val_local_idx = split_train_val_by_subtype_strict(
        X_train_val, y_train_val, groups_train_val,
        min_ratio=VAL_RATIO_MIN,
        max_ratio=VAL_RATIO_MAX,
        min_classes=3,
        random_state=run + 200,
        max_attempts=500
    )

    if train_local_idx is None:
        print(f"⚠️ 无法在严格范围内找到满足三类要求的验证集划分，跳过 Run {run+1}")
        skipped_runs += 1
        continue

    X_train = X_train_val.iloc[train_local_idx]
    y_train = y_train_val.iloc[train_local_idx]
    X_val = X_train_val.iloc[val_local_idx]
    y_val = y_train_val.iloc[val_local_idx]

    actual_val_ratio_trainval = len(X_val) / len(X_train_val)
    print(f"Train size: {len(X_train)} (Subtypes: {groups_train_val.iloc[train_local_idx].nunique()})")
    print(f"Val size: {len(X_val)} (Subtypes: {groups_train_val.iloc[val_local_idx].nunique()})")
    print(f"验证集占 train_val 比例: {actual_val_ratio_trainval:.4f}")
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Val label distribution: {y_val.value_counts().to_dict()}")

    # =====================================================
    # 在训练+验证合并集上，按 one-vs-rest 找最优阈值
    # =====================================================
    X_train_val_combined = pd.concat([X_train, X_val], axis=0)
    y_train_val_combined = pd.concat([y_train, y_val], axis=0)

    scores_train_val = X_train_val_combined["Marker_Count"].values
    scores_test = X_test["Marker_Count"].values

    print("\n--- 在训练+验证集上寻找各类别最优阈值 ---")

    run_result = {"Run": run + 1}
    test_auroc_list = []

    for class_id in [0, 1, 2]:
        y_binary_train = (y_train_val_combined == class_id).astype(int).values
        y_binary_test = (y_test == class_id).astype(int).values

        # 训练+验证集上计算 ROC
        fpr, tpr, thresholds = roc_curve(y_binary_train, scores_train_val)
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

        # 测试集评估
        y_pred_test = (scores_test >= best_threshold).astype(int)

        # 只有当测试集中该类别样本数 > 0 时才能计算 AUROC
        if y_binary_test.sum() > 0 and y_binary_test.sum() < len(y_binary_test):
            test_auroc = roc_auc_score(y_binary_test, scores_test)
            test_auprc = average_precision_score(y_binary_test, scores_test)
        else:
            test_auroc = np.nan
            test_auprc = np.nan

        # 二分类指标
        test_acc = accuracy_score(y_binary_test, y_pred_test)
        test_prec = precision_score(y_binary_test, y_pred_test, zero_division=0)
        test_rec = recall_score(y_binary_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_binary_test, y_pred_test, zero_division=0)
        test_mcc = matthews_corrcoef(y_binary_test, y_pred_test)

        print(f"\n类别 {class_names[class_id]} (one-vs-rest):")
        print(f"  训练+验证集 AUROC: {auroc_train_val:.4f}")
        print(f"  最优阈值: {best_threshold:.4f}")
        print(f"  测试集 AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
        print(f"  测试集 ACC: {test_acc:.4f}, F1: {test_f1:.4f}, MCC: {test_mcc:.4f}")

        run_result[f"{class_names[class_id]}_Threshold"] = best_threshold
        run_result[f"{class_names[class_id]}_TrainVal_AUROC"] = auroc_train_val
        run_result[f"{class_names[class_id]}_Test_AUROC"] = test_auroc
        run_result[f"{class_names[class_id]}_Test_AUPRC"] = test_auprc
        run_result[f"{class_names[class_id]}_Test_ACC"] = test_acc
        run_result[f"{class_names[class_id]}_Test_Precision"] = test_prec
        run_result[f"{class_names[class_id]}_Test_Recall"] = test_rec
        run_result[f"{class_names[class_id]}_Test_F1"] = test_f1
        run_result[f"{class_names[class_id]}_Test_MCC"] = test_mcc

        if not np.isnan(test_auroc):
            test_auroc_list.append(test_auroc)

    # 各类别平均 AUROC 作为整体指标
    if test_auroc_list:
        run_result["Test_macro_AUROC"] = np.mean(test_auroc_list)
    else:
        run_result["Test_macro_AUROC"] = np.nan

    all_results.append(run_result)

# =========================================================
# 6. 汇总结果
# =========================================================
if len(all_results) == 0:
    print("\n" + "=" * 100)
    print("❌ 所有 Run 均因验证集划分失败而被跳过，无结果保存。")
    print("=" * 100)
else:
    results_df = pd.DataFrame(all_results)
    summary_mean = results_df.mean(numeric_only=True)
    summary_std = results_df.std(numeric_only=True)

    print("\n" + "=" * 100)
    print("基于标志物数目的阈值分类结果（均值 ± 标准差）")
    print("=" * 100)

    print("\n各类别最优阈值:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_Threshold']:.4f} ± {summary_std[f'{class_name}_Threshold']:.4f}")

    print("\n训练+验证集 AUROC:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_TrainVal_AUROC']:.4f} ± {summary_std[f'{class_name}_TrainVal_AUROC']:.4f}")

    print("\n测试集 AUROC:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_Test_AUROC']:.4f} ± {summary_std[f'{class_name}_Test_AUROC']:.4f}")

    print("\n测试集各项指标 (macro AUROC):")
    print(f"  macro_AUROC: {summary_mean['Test_macro_AUROC']:.4f} ± {summary_std['Test_macro_AUROC']:.4f}")

    print("\n各类别测试集 AUPRC:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_Test_AUPRC']:.4f} ± {summary_std[f'{class_name}_Test_AUPRC']:.4f}")

    print("\n各类别测试集 F1:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_Test_F1']:.4f} ± {summary_std[f'{class_name}_Test_F1']:.4f}")

    print("\n各类别测试集 MCC:")
    for class_name in class_names.values():
        print(f"  {class_name}: {summary_mean[f'{class_name}_Test_MCC']:.4f} ± {summary_std[f'{class_name}_Test_MCC']:.4f}")

    # 保存
    results_df.to_csv(f"{OUTPUT_DIR}/Threshold_MarkerCount_Results.csv", index=False)
    summary_mean.to_csv(f"{OUTPUT_DIR}/Summary_Mean.csv")
    summary_std.to_csv(f"{OUTPUT_DIR}/Summary_STD.csv")

    print("\n结果已保存至目录:", OUTPUT_DIR)

print(f"\n总 Run 数: {len(final_splits)}, 成功 {len(final_splits)-skipped_runs} 轮，跳过 {skipped_runs} 轮。")