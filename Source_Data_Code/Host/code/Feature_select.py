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
from xgboost import XGBClassifier

# ===== 设置全局随机种子 =====
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
VAL_RATIO_MIN = 0.15
VAL_RATIO_MAX = 0.25

# =========================================================
# 2. 文件路径
# =========================================================
file_path = "../data/host_markers_with_cluster.csv"
output_dir = "TOPN_XGB_RF_Importance_NoCV_3class_force3"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_csv(file_path)

label_map = {"Human": 0, "Avian": 1, "Swine": 2}
df["Host_class"] = df["Host_class"].map(label_map)
if df["Host_class"].isna().any():
    raise ValueError("存在未映射的标签值，请检查 label_map 是否涵盖所有类别。")

X = df.drop(columns=["Strain", "Subtype", "Host_class", "Cluster", "Isolate_Id", "Host", "marker_count"], errors="ignore")
y = df["Host_class"]
groups = df["Cluster"]

print("="*100)
print("Feature Shape:", X.shape)
print("Unique Subtypes:", groups.nunique())
print("标签类别数:", len(np.unique(y)))
print("标签分布:", y.value_counts().to_dict())
print("="*100)

# =========================================================
# 4. 多分类评估函数
# =========================================================
def evaluate(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
        "AUPRC": average_precision_score(y_true, y_prob, average='macro')
    }

# =========================================================
# 5. TopN范围（所有特征）
# =========================================================
TOPN_LIST = list(range(1, X.shape[1] + 1))
print(f"测试的 TopN 值: {TOPN_LIST[:10]} ... 共 {len(TOPN_LIST)} 个")

# =========================================================
# 6. 生成候选测试集（强制训练/测试集三类齐全）
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
        if len(np.unique(y.iloc[train_idx])) < 3 or len(np.unique(y.iloc[test_idx])) < 3:
            continue
        if len(np.unique(groups.iloc[train_idx])) < 2:
            continue
        candidate_splits.append(test_idx)
        if len(candidate_splits) >= N_CANDIDATES:
            break

print(f"生成候选测试集数量: {len(candidate_splits)}")

# =========================================================
# 7. 筛选低重叠测试集（Jaccard ≤ 0.8）
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
if len(final_splits) == 0:
    raise RuntimeError("未能选出任何满足条件的测试集，请尝试放宽比例范围或增加 N_CANDIDATES。")

# =========================================================
# 8. 划分函数：严格模式（验证集比例基于 train_val，不尝试放宽）
# 加入 Cluster 前先判断，超上限则跳过该 Cluster。
# 若严格范围内无法满足条件，返回 (None, None)。
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

            # 若加入后超上限，则跳过该 Cluster（不加入）
            if new_ratio > max_ratio:
                continue

            cumsum += len(idx)
            val_groups.append(g)

            # 检查是否达到下限且三类齐全
            if cumsum / total >= min_ratio:
                val_idx = np.concatenate([np.where(groups == g)[0] for g in val_groups])
                train_idx = np.setdiff1d(np.arange(total), val_idx)
                if (len(np.unique(y.iloc[train_idx])) >= min_classes and
                    len(np.unique(y.iloc[val_idx])) >= min_classes):
                    return train_idx, val_idx

    return None, None

# =========================================================
# 9. 外层循环
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
    print("Test ratio:", len(test_idx)/len(X))

    # 严格模式划分验证集（比例基于 train_val）
    train_local_idx, val_local_idx = split_train_val_by_subtype_strict(
        X_train_val, y_train_val, groups_train_val,
        min_ratio=VAL_RATIO_MIN,
        max_ratio=VAL_RATIO_MAX,
        min_classes=3,
        random_state=run+200,
        max_attempts=500
    )

    if train_local_idx is None:
        print(f"⚠️ 无法在严格范围 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}] 内找到满足三类要求的验证集划分，跳过 Run {run+1}")
        skipped_runs += 1
        continue

    X_train = X_train_val.iloc[train_local_idx]
    y_train = y_train_val.iloc[train_local_idx]
    X_val = X_train_val.iloc[val_local_idx]
    y_val = y_train_val.iloc[val_local_idx]

    actual_val_ratio = len(X_val) / len(X_train_val)
    print(f"Train size: {len(X_train)} (Subtypes: {groups_train_val.iloc[train_local_idx].nunique()})")
    print(f"Val size: {len(X_val)} (Subtypes: {groups_train_val.iloc[val_local_idx].nunique()})")
    print(f"验证集占 train_val 比例: {actual_val_ratio:.4f} (严格范围 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}])")
    print(f"验证集占全体比例: {len(X_val)/len(X):.4f}")

    # =================================================
    # 特征重要性（RF）
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
    # TopN循环（XGBoost）
    # =================================================
    for top_n in TOPN_LIST:
        selected_features = ranked_features[:top_n]

        final_model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=123,
            tree_method='hist',
            device='cuda'
        )
        final_model.fit(X_train[selected_features], y_train)

        y_val_prob = final_model.predict_proba(X_val[selected_features])
        val_metrics = evaluate(y_val, y_val_prob)


        all_results.append({
            "Run": run+1,
            "TopN": top_n,
            **{f"Val_{k}": v for k, v in val_metrics.items()},
        })

# =========================================================
# 10. 汇总与保存
# =========================================================
if all_results:
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
else:
    print("\n" + "="*100)
    print("❌ 所有 Run 均因验证集划分失败而被跳过，无结果保存。")
    print("="*100)

print("\n" + "="*100)
print(f"总 Run 数: {len(final_splits)}, 成功 {len(final_splits)-skipped_runs} 轮，跳过 {skipped_runs} 轮。")
print("TopN Feature Selection Finished (RF Importance + XGB Classifier, No CV, 3-class)")
print("="*100)