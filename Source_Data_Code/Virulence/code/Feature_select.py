import pandas as pd
import numpy as np
import warnings
import random

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(123)
random.seed(123)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
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

# 特征矩阵（删除非特征列）
X = df.drop(columns=["Strain", "Subtype", "label", "id", "subtype", "Cluster", "cluster"], errors="ignore")
y = df["label"]
groups = df["cluster"]   # 使用 cluster 列

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Clusters:", groups.nunique())

# =========================================================
# 3. 评估函数
# =========================================================
def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob)
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
    candidates = []   # 存储 (train_idx, val_idx, test_idx, test_clusters)

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
    # 按测试 Cluster 的 Jaccard 重叠筛选
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

# =========================================================
# 5. 主循环：对每个划分，计算RF重要性，遍历TopN特征
# =========================================================
all_results = []

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

    # ---- 计算 RF 基尼重要性 ----
    rf = RandomForestClassifier(n_estimators=100, random_state=27, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(importance)[::-1]
    ranked_features = feature_names[sorted_idx].tolist()
    total_features = len(ranked_features)

    # ---- 遍历 TopN ----
    for top_n in range(1, total_features + 1):
        selected_features = ranked_features[:top_n]
        X_train_sel = X_train[selected_features]
        X_val_sel = X_val[selected_features]
        X_test_sel = X_test[selected_features]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # 训练 xgb（默认参数，可加class_weight='balanced'）
        xgb = XGBClassifier(eval_metric='logloss', random_state=123)
        xgb.fit(X_train_scaled, y_train)

        # 验证集评估
        y_val_prob = xgb.predict_proba(X_val_scaled)[:, 1]
        val_metrics = evaluate(y_val, y_val_prob)


        # 记录结果
        result = {
            "Run": run + 1,
            "TopN": top_n,
            "Val_ACC": val_metrics["ACC"],
            "Val_Precision": val_metrics["Precision"],
            "Val_Recall": val_metrics["Recall"],
            "Val_F1": val_metrics["F1"],
            "Val_MCC": val_metrics["MCC"],
            "Val_AUROC": val_metrics["AUROC"],
            "Val_AUPRC": val_metrics["AUPRC"],
        }
        all_results.append(result)

        if top_n % 50 == 0:
            print(f"  TopN {top_n}/{total_features} done")

# =========================================================
# 6. 汇总结果
# =========================================================
results_df = pd.DataFrame(all_results)
print("\n" + "=" * 100)
print("All Results (每个 Run 和 TopN)")
print("=" * 100)
print(results_df)

# 按 TopN 汇总均值和标准差
summary_mean = results_df.groupby("TopN").mean()
summary_std = results_df.groupby("TopN").std()

print("\n" + "=" * 100)
print("Mean Results by TopN")
print("=" * 100)
print(summary_mean)

print("\n" + "=" * 100)
print("Std Results by TopN")
print("=" * 100)
print(summary_std)

# 保存结果
results_df.to_csv("XGB_TopN_RF_Importance_Results.csv", index=False)
summary_mean.to_csv("XGB_TopN_Mean.csv")
summary_std.to_csv("XGB_TopN_STD.csv")

print("\n" + "=" * 100)
print("文件已保存：")
print("  - XGB_TopN_RF_Importance_Results.csv : 详细结果")
print("  - XGB_TopN_Mean.csv                  : 各 TopN 的指标均值")
print("  - XGB_TopN_STD.csv                   : 各 TopN 的指标标准差")
print("=" * 100)