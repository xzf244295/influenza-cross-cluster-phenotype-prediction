import pandas as pd
import numpy as np
import warnings
import random

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(27)
random.seed(27)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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

X = df.drop(columns=["Strain", "Subtype", "label"], errors="ignore")
y = df["label"]
groups = df["Subtype"]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Subtypes:", groups.nunique())

# =========================================================
# 3. 定义模型（默认参数）
# =========================================================
models = {
    "LR": LogisticRegression(max_iter=1000, random_state=123),
    "SVM": SVC(probability=True, random_state=123),
    "KNN": KNeighborsClassifier(),
    "GNB": GaussianNB(),
    "RF": RandomForestClassifier(random_state=123),
    "XGB": XGBClassifier(eval_metric='logloss', random_state=123),
    "GBDT": GradientBoostingClassifier(random_state=123)
}

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
# 5. 候选测试集生成与筛选（基于 Subtype，重叠率 ≤ 0.8）
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

print("生成候选测试集...")
candidates = generate_candidate_splits(X, y, groups, N_CANDIDATES, TEST_RATIO_MIN, TEST_RATIO_MAX)
print(f"候选数量: {len(candidates)}")
final_splits = select_final_splits(candidates, N_FINAL_SPLITS, MAX_OVERLAP, groups)
print(f"最终选中 {len(final_splits)} 个独立测试集")

# =========================================================
# 6.训练/验证划分函数
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
# 7. 外层循环（对每个独立测试集）
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

    # 调用改进后的划分函数，强制验证集占 train_val 比例在 [VAL_RATIO_MIN, VAL_RATIO_MAX]
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
    # 对每个模型训练（无交叉验证）
    # =====================================================
    for model_name, model in models.items():
        print("\n" + "#" * 100)
        print(f"Model: {model_name}")
        print("#" * 100)

        # 标准化（对需要标准化的模型）
        scaler = StandardScaler()
        if model_name in ["LR", "SVM", "KNN"]:
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            X_tr = X_train_scaled
            X_vl = X_val_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_vl = X_val
            X_te = X_test

        # 训练模型
        model.fit(X_tr, y_train)

        # 验证集评估
        y_val_prob = model.predict_proba(X_vl)[:, 1]
        val_metrics = evaluate(y_val, y_val_prob)

        print("\nValidation Result:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")


        # 保存结果
        result = {
            "Run": run + 1,
            "Model": model_name,
            "Val_ACC": val_metrics["ACC"],
            "Val_Precision": val_metrics["Precision"],
            "Val_Recall": val_metrics["Recall"],
            "Val_F1": val_metrics["F1"],
            "Val_MCC": val_metrics["MCC"],
            "Val_AUROC": val_metrics["AUROC"],
            "Val_AUPRC": val_metrics["AUPRC"],
        }
        all_results.append(result)

# =========================================================
# 8. 汇总结果
# =========================================================
if len(all_results) == 0:
    print("\n" + "=" * 100)
    print("❌ 所有轮次均无法找到满足验证集比例要求的划分，无结果保存。")
    print("=" * 100)
else:
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 100)
    print("All Results")
    print("=" * 100)
    print(results_df)

    summary_mean = results_df.groupby("Model").mean(numeric_only=True)
    summary_std = results_df.groupby("Model").std(numeric_only=True)

    print("\nMean Results")
    print(summary_mean)
    print("\nStd Results")
    print(summary_std)

    results_df.to_csv("all_random_split_results_no_cv.csv", index=False)
    summary_mean.to_csv("summary_mean_no_cv.csv")
    summary_std.to_csv("summary_std_no_cv.csv")

    best_model = summary_mean["Val_AUROC"].idxmax()
    print("\n" + "=" * 100)
    print(f"Best Model Based On Validation MCC: {best_model}")
    print("=" * 100)

print("\n" + "=" * 100)
print(f"实验结束。成功运行 {len(all_results)//len(models) if len(models)>0 else 0} 轮，跳过 {skipped_runs} 轮。")
print("=" * 100)