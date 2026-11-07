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


# ===== 设置全局随机种子，保证结果可复现 =====
np.random.seed(27)
random.seed(27)

warnings.filterwarnings("ignore")

# =========================================================
# 1. 参数
# =========================================================
N_CANDIDATES = 1000          # 候选尝试次数
TEST_RATIO_MIN = 0.15
TEST_RATIO_MAX = 0.25
MAX_OVERLAP = 0.8
N_FINAL_SPLITS = 10
VAL_RATIO_MIN = 0.15         # 验证集占 train_val 的比例下限
VAL_RATIO_MAX = 0.25         # 验证集占 train_val 的比例上限

# =========================================================
# 2. 文件路径
# =========================================================
file_path = "../data/all_binding_marker_matrix_with_subtype.csv"
output_dir = "TOPN_RF_SVM_Importance_NoCV"
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
# 5. TopN范围（所有特征）
# =========================================================
TOPN_LIST = list(range(1, X.shape[1] + 1))

# =========================================================
# 6. 生成候选测试集（基于Subtype，比例在0.15~0.25）
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
# 7. 筛选低重叠测试集（Jaccard重叠率 ≤ 0.8）
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
# 8. 从 train_val 中划分训练集和验证集（按Subtype）
#    强制验证集占 train_val 的比例在 [min_ratio, max_ratio] 内
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
    total = len(X)  # 这里是 train_val 的样本数

    for trial in range(max_trials):
        shuffled = rng.permutation(unique_groups)
        cumsum = 0
        val_groups = []
        for g in shuffled:
            idx = np.where(groups == g)[0]
            new_cumsum = cumsum + len(idx)
            new_ratio = new_cumsum / total
            # 如果加入该亚型会超过上限，则跳过该亚型
            if new_ratio > max_ratio:
                continue
            # 加入该亚型
            cumsum = new_cumsum
            val_groups.append(g)
            current_ratio = cumsum / total
            if current_ratio >= min_ratio:
                # 达到下限，检查类别
                val_idx = np.concatenate([np.where(groups == g)[0] for g in val_groups])
                train_idx = np.setdiff1d(np.arange(total), val_idx)
                if (len(np.unique(y.iloc[train_idx])) >= min_classes and
                    len(np.unique(y.iloc[val_idx])) >= min_classes):
                    return train_idx, val_idx
        # 如果本次排列未找到，继续下一次随机排列
    return None, None

# =========================================================
# 9. 外层实验（无交叉验证）
# =========================================================
all_results = []
skipped_runs = 0

for run, test_idx in enumerate(final_splits):
    print("\n" + "="*100)
    print(f"Random Split Run {run+1}")
    print("="*100)

    train_val_idx = np.setdiff1d(np.arange(len(X)), test_idx)

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("Train-Val Subtypes:", len(groups_train_val.unique()))
    print("Test Subtypes:", len(groups.iloc[test_idx].unique()))
    print("Test ratio (占整体):", len(test_idx)/len(X))

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

    # =================================================
    # Step1: 使用随机森林计算特征重要性（基尼重要性）
    # =================================================
    rf_importance = RandomForestClassifier(random_state=123, n_jobs=-1)
    rf_importance.fit(X_train, y_train)
    importance = rf_importance.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
    ranked_features = importance_df["Feature"].tolist()

    # =================================================
    # Step2: TopN循环
    # =================================================
    for top_n in TOPN_LIST:
        selected_features = ranked_features[:top_n]

        # SVM 对特征尺度敏感，训练前做标准化（用训练集拟合 scaler）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        X_val_scaled = scaler.transform(X_val[selected_features])
        X_test_scaled = scaler.transform(X_test[selected_features])

        final_model = SVC(probability=True, random_state=123)
        final_model.fit(X_train_scaled, y_train)

        # 验证集评估
        y_val_prob = final_model.predict_proba(X_val_scaled)[:, 1]
        val_metrics = evaluate(y_val, y_val_prob)

        all_results.append({
            "Run": run+1,
            "TopN": top_n,
            **{f"Val_{k}": v for k, v in val_metrics.items()},
        })

# =========================================================
# 10. 汇总与保存
# =========================================================
if len(all_results) == 0:
    print("\n" + "="*100)
    print("❌ 所有轮次均无法找到满足验证集比例要求的划分，无结果保存。")
    print("="*100)
else:
    results_df = pd.DataFrame(all_results)
    summary_df = results_df.groupby("TopN").mean(numeric_only=True).reset_index()
    std_df = results_df.groupby("TopN").std(numeric_only=True).reset_index()

    results_df.to_csv(f"{output_dir}/All_TopN_Results.csv", index=False)
    summary_df.to_csv(f"{output_dir}/TopN_Mean_Results.csv", index=False)
    std_df.to_csv(f"{output_dir}/TopN_STD_Results.csv", index=False)

    best_row = summary_df.loc[summary_df["Val_AUROC"].idxmax()]
    print("\n" + "="*100)
    print("Best TopN (by Val AUROC)")
    print("="*100)
    print(best_row)

print("\n" + "="*100)
print(f"TopN Feature Selection Finished (RF Importance + SVM Classifier, No CV). "
      f"成功运行 {len(all_results)//len(TOPN_LIST) if len(TOPN_LIST)>0 else 0} 轮，跳过 {skipped_runs} 轮。")
print("="*100)