import pandas as pd
import numpy as np
import warnings
import os
import random
import joblib  # 用于保存模型

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
VAL_RATIO_MIN = 0.15
VAL_RATIO_MAX = 0.25

# 固定特征数量
TOP_N = 79

# XGBoost GPU 加速参数
XGB_GPU_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
}

# XGBoost 网格参数（多分类）
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# =========================================================
# 2. 文件路径
# =========================================================
file_path = "../data/host_markers_with_cluster.csv"
output_dir = "XGB_GPU_Top79_GridSearch_NoCV_3class_with_save"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)
feature_dir = os.path.join(output_dir, "features")
os.makedirs(feature_dir, exist_ok=True)

# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_csv(file_path)

label_map = {"Human": 0, "Avian": 1, "Swine": 2}
df["Host_class"] = df["Host_class"].map(label_map)

X = df.drop(columns=["Strain", "Subtype", "Host_class", "Host", "Isolate_Id", "Cluster", "marker_count"], errors="ignore")
y = df["Host_class"]
groups = df["Cluster"]

print("="*100)
print("Feature Shape:", X.shape)
print("Unique Subtypes:", groups.nunique())
print("标签类别数:", len(np.unique(y)))
print("标签分布:", y.value_counts().to_dict())
print("固定特征数量:", TOP_N)
print("XGB 网格参数:", XGB_PARAM_GRID)
print("XGB GPU 参数:", XGB_GPU_PARAMS)
print("="*100)

# =========================================================
# 4. 多分类评估函数
# =========================================================
def evaluate(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
        "AUPRC": average_precision_score(y_true, y_prob, average='macro')
    }
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i in range(3):
        metrics[f"Precision_class{i}"] = per_class_precision[i]
        metrics[f"Recall_class{i}"] = per_class_recall[i]
        metrics[f"F1_class{i}"] = per_class_f1[i]
    return metrics

# =========================================================
# 5. 生成候选测试集（基于Subtype，强制三类齐全）
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

if len(final_splits) == 0:
    raise RuntimeError("未能选出任何满足条件的测试集，请尝试放宽比例范围或增加 N_CANDIDATES。")

# =========================================================
# 7. 【严格模式】划分函数：加入前判断，绝不超上限
# =========================================================
def split_train_val_by_subtype_strict(X, y, groups, min_ratio, max_ratio,
                                      min_classes=3, random_state=None, max_attempts=500):
    """
    严格将验证集比例控制在 [min_ratio, max_ratio] 内（基于 train_val 的样本数）。
    加入 Cluster 前先判断，超上限则跳过该 Cluster。
    若所有尝试均失败，返回 (None, None)。
    """
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
# 8. 外层实验（无交叉验证，固定TopN + 网格搜索 XGB GPU）
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

    # ---- 严格模式划分验证集（比例基于 train_val） ----
    train_local_idx, val_local_idx = split_train_val_by_subtype_strict(
        X_train_val, y_train_val, groups_train_val,
        min_ratio=VAL_RATIO_MIN,
        max_ratio=VAL_RATIO_MAX,
        min_classes=3,
        random_state=run+200,
        max_attempts=500
    )

    if train_local_idx is None:
        print(f"⚠️ 无法在严格范围 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}] 内找到满足三类要求的划分，跳过 Run {run+1}")
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
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Val label distribution: {y_val.value_counts().to_dict()}")

    # =================================================
    # Step1: 在训练集上计算特征重要性，用于网格搜索
    # =================================================
    rf_importance = RandomForestClassifier(random_state=123, n_jobs=-1)
    rf_importance.fit(X_train, y_train)
    importance = rf_importance.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
    ranked_features = importance_df["Feature"].tolist()
    selected_features = ranked_features[:TOP_N]

    # 保存训练集选出的特征（可选）
    feature_file = os.path.join(feature_dir, f"run_{run+1}_top{TOP_N}_features_train.csv")
    pd.Series(selected_features).to_csv(feature_file, index=False, header=False)
    print(f"训练集特征列表已保存至: {feature_file}")

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]

    # =================================================
    # Step2: 网格搜索（验证集 AUROC 为指标）—— XGBoost GPU
    # =================================================
    best_params = None
    best_val_auc = -np.inf

    for params in ParameterGrid(XGB_PARAM_GRID):
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=123,
            n_jobs=-1,
            **XGB_GPU_PARAMS,
            **params
        )
        model.fit(X_train_sel, y_train)
        y_val_prob = model.predict_proba(X_val_sel)
        auc = roc_auc_score(y_val, y_val_prob, multi_class='ovr', average='macro')
        if auc > best_val_auc:
            best_val_auc = auc
            best_params = params

    print(f"Best Params: {best_params}")
    print(f"Best Val AUROC (macro-ovr): {best_val_auc:.4f}")

    # =================================================
    # Step3: 在合并集上重新选择特征，并训练最终模型
    # =================================================
    # 合并原始训练和验证集
    X_train_val_combined = pd.concat([X_train, X_val], axis=0)
    y_train_val_combined = pd.concat([y_train, y_val], axis=0)

    # 在合并集上重新选择特征
    rf_importance_final = RandomForestClassifier(random_state=123, n_jobs=-1)
    rf_importance_final.fit(X_train_val_combined, y_train_val_combined)
    importance_final = rf_importance_final.feature_importances_
    importance_df_final = pd.DataFrame({
        "Feature": X_train_val_combined.columns,
        "Importance": importance_final
    }).sort_values(by="Importance", ascending=False)
    ranked_features_final = importance_df_final["Feature"].tolist()
    selected_features_final = ranked_features_final[:TOP_N]

    # 保存最终选出的特征
    feature_file_final = os.path.join(feature_dir, f"run_{run+1}_top{TOP_N}_features_final.csv")
    pd.Series(selected_features_final).to_csv(feature_file_final, index=False, header=False)
    print(f"合并集最终特征列表已保存至: {feature_file_final}")

    # 应用新特征
    X_train_val_final_sel = X_train_val_combined[selected_features_final]
    X_test_final_sel = X_test[selected_features_final]

    # 用最佳参数训练最终模型
    final_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=123,
        n_jobs=-1,
        **XGB_GPU_PARAMS,
        **best_params
    )
    final_model.fit(X_train_val_final_sel, y_train_val_combined)

    model_file = os.path.join(model_dir, f"run_{run+1}_XGB_model.joblib")
    joblib.dump(final_model, model_file)
    print(f"模型已保存至: {model_file}")

    # 测试集评估
    y_test_prob = final_model.predict_proba(X_test_final_sel)
    test_metrics = evaluate(y_test, y_test_prob)

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {
        "Run": run + 1,
        "TopN": TOP_N,
        "Best_Params": str(best_params),
        "Best_Val_AUROC": best_val_auc,
        **{f"Test_{k}": v for k, v in test_metrics.items()}
    }
    all_results.append(result)

# =========================================================
# 9. 汇总与保存
# =========================================================
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/all_runs_results.csv", index=False)

    summary_cols = [c for c in results_df.columns if c.startswith("Test_") and results_df[c].dtype in ['float64', 'int64']]
    summary = results_df[summary_cols].agg(['mean', 'std'])
    summary.to_csv(f"{output_dir}/test_summary.csv")

    print("\n" + "="*100)
    print("所有 Run 完成！")
    print(f"结果汇总已保存至 {output_dir}")
    print("="*100)
    print("\n测试集指标均值 ± 标准差:")
    print(summary)

    print("\n最佳参数分布:")
    print(results_df["Best_Params"].value_counts())
else:
    print("\n" + "="*100)
    print("❌ 所有 Run 均因验证集划分失败而被跳过，无结果保存。")
    print("="*100)

print(f"\n总 Run 数: {len(final_splits)}, 成功 {len(final_splits)-skipped_runs} 轮，跳过 {skipped_runs} 轮。")
print("XGBoost (GPU) TopN 特征选择 + 网格搜索 (No CV, 3-class) 完成。")

# =========================================================
# 10. 测试集类别数量统计
# =========================================================
print("\n" + "="*100)
print("测试集类别数量统计（10次划分）")
print("="*100)

counts_list = []
for run, test_idx in enumerate(final_splits):
    y_test_subset = y.iloc[test_idx]
    count_dict = y_test_subset.value_counts().to_dict()
    c0 = count_dict.get(0, 0)
    c1 = count_dict.get(1, 0)
    c2 = count_dict.get(2, 0)
    counts_list.append([c0, c1, c2])
    print(f"Run {run+1:2d}: class0={c0:4d}, class1={c1:4d}, class2={c2:4d}")

counts_array = np.array(counts_list)
mean_counts = counts_array.mean(axis=0)
std_counts = counts_array.std(axis=0, ddof=1)

print("\n" + "-"*100)
print(f"均值  : class0={mean_counts[0]:.1f}, class1={mean_counts[1]:.1f}, class2={mean_counts[2]:.1f}")
print(f"标准差: class0={std_counts[0]:.1f}, class1={std_counts[1]:.1f}, class2={std_counts[2]:.1f}")
print("="*100)