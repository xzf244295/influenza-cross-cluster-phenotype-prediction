import pandas as pd
import numpy as np
import warnings
import random
import os

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(123)
random.seed(123)

from sklearn.preprocessing import StandardScaler
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
TOP_N = 51                    # 固定特征数量

# 输出目录
OUTPUT_DIR = "XGB_Top51_GridSearch_ClusterSplit_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGBoost 网格参数

XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.2],
    'scale_pos_weight': [1, 2]   
}
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

# =========================================================
# 5. 主循环：对每个划分，特征选择 + 网格搜索 + 测试
# =========================================================
all_test_results = []        # 存储测试指标
all_run_details = []         # 存储详细结果（最优参数等）

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

    # ---- 1. RF 基尼重要性，选出 Top51 特征 ----
    rf = RandomForestClassifier(n_estimators=100, random_state=27, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(importance)[::-1]
    ranked_features = feature_names[sorted_idx].tolist()
    selected_features = ranked_features[:TOP_N]

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    # ---- 2. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # ---- 3. 网格搜索（以验证集 AUROC 最大为目标） ----
    best_auc = -1
    best_params = None
    for n_estimators in XGB_PARAM_GRID['n_estimators']:
        for max_depth in XGB_PARAM_GRID['max_depth']:
            for learning_rate in XGB_PARAM_GRID['learning_rate']:
                for subsample in XGB_PARAM_GRID['subsample']:
                    for colsample_bytree in XGB_PARAM_GRID['colsample_bytree']:
                        for gamma in XGB_PARAM_GRID['gamma']:
                            for scale_pos_weight in XGB_PARAM_GRID['scale_pos_weight']:
                                try:
                                    xgb = XGBClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        subsample=subsample,
                                        colsample_bytree=colsample_bytree,
                                        gamma=gamma,
                                        scale_pos_weight=scale_pos_weight,
                                        eval_metric='logloss',
                                        use_label_encoder=False,
                                        random_state=123
                                    )
                                    xgb.fit(X_train_scaled, y_train)
                                    y_val_prob = xgb.predict_proba(X_val_scaled)[:, 1]
                                    val_auc = roc_auc_score(y_val, y_val_prob)
                                    if val_auc > best_auc:
                                        best_auc = val_auc
                                        best_params = {
                                            'n_estimators': n_estimators,
                                            'max_depth': max_depth,
                                            'learning_rate': learning_rate,
                                            'subsample': subsample,
                                            'colsample_bytree': colsample_bytree,
                                            'gamma': gamma,
                                            'scale_pos_weight': scale_pos_weight
                                        }
                                except Exception as e:
                                    continue

    print(f"最优参数: {best_params}, 验证集 AUROC: {best_auc:.4f}")

    # ---- 4. 用最优参数在训练集上训练最终模型 ----
    final_xgb = XGBClassifier(**best_params, eval_metric='logloss', use_label_encoder=False, random_state=123)
    final_xgb.fit(X_train_scaled, y_train)

    # ---- 5. 测试集评估 ----
    y_test_prob = final_xgb.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate(y_test, y_test_prob)

    print("\n测试集结果:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- 6. 保存结果 ----
    test_record = {
        "Run": run + 1,
        "Best_n_estimators": best_params['n_estimators'],
        "Best_max_depth": best_params['max_depth'],
        "Best_learning_rate": best_params['learning_rate'],
        "Best_subsample": best_params['subsample'],
        "Best_colsample_bytree": best_params['colsample_bytree'],
        "Best_gamma": best_params['gamma'],
        "Best_scale_pos_weight": best_params['scale_pos_weight'],
        "Best_Val_AUROC": best_auc,
        **{f"Test_{k}": v for k, v in test_metrics.items()}
    }
    all_test_results.append(test_record)
    all_run_details.append(test_record)

# =========================================================
# 6. 汇总测试结果
# =========================================================
results_df = pd.DataFrame(all_test_results)
summary_mean = results_df.mean(numeric_only=True)
summary_std = results_df.std(numeric_only=True)

print("\n" + "=" * 100)
print("10次测试集结果汇总（均值 ± 标准差）")
print("=" * 100)
for metric in ["Test_ACC", "Test_Precision", "Test_Recall", "Test_F1", "Test_MCC", "Test_AUROC", "Test_AUPRC"]:
    print(f"{metric}: {summary_mean[metric]:.4f} ± {summary_std[metric]:.4f}")

print("\n最优参数分布:")
print(results_df[['Best_n_estimators', 'Best_max_depth', 'Best_learning_rate',
                  'Best_subsample', 'Best_colsample_bytree', 'Best_gamma',
                  'Best_scale_pos_weight']].value_counts())

# 保存结果
results_df.to_csv(f"{OUTPUT_DIR}/XGB_Top51_GridSearch_Test_Results.csv", index=False)
summary_mean.to_csv(f"{OUTPUT_DIR}/Summary_Mean.csv")
summary_std.to_csv(f"{OUTPUT_DIR}/Summary_STD.csv")

print("\n结果已保存至目录:", OUTPUT_DIR)
print("=" * 100)