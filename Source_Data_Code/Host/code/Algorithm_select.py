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
VAL_RATIO_MIN = 0.15
VAL_RATIO_MAX = 0.25

# =========================================================
# 2. 读取数据
# =========================================================
file_path = "../data/host_markers_with_cluster"
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

X = df.drop(columns=["Strain", "Subtype", "Host_class", "Host", "Isolate_Id", "Cluster", "marker_count"], errors="ignore")
y = df["Host_class"]
groups = df["Cluster"]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("Unique Subtypes:", groups.nunique())

# =========================================================
# 3. 定义模型（默认参数，均支持多分类）
# =========================================================
models = {
    "LR": LogisticRegression(max_iter=1000, random_state=123),
    "SVM": SVC(probability=True, random_state=123),
    "KNN": KNeighborsClassifier(),
    "GNB": GaussianNB(),
    "RF": RandomForestClassifier(random_state=123),
    "XGB": XGBClassifier(eval_metric='mlogloss', random_state=123),
    "GBDT": GradientBoostingClassifier(random_state=123)
}

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

    # =====================================================
    # 内层：严格比例范围的验证集划分（不尝试放宽）
    # 加入 Cluster 前先判断加入后是否超上限，超上限则跳过该 Cluster。
    # 若严格范围内无法满足条件，则返回 None 并跳过该 Run。
    # =====================================================
    def split_train_val_by_subtype_strict(X, y, groups, min_ratio, max_ratio,
                                          min_classes=2, random_state=None, max_attempts=500):
        """
        严格将验证集比例控制在 [min_ratio, max_ratio] 内。
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

                # 若加入后超上限，则跳过该 Cluster（不加入）
                if new_ratio > max_ratio:
                    continue

                cumsum += len(idx)
                val_groups.append(g)

                # 检查是否达到下限且类别齐全
                if cumsum / total >= min_ratio:
                    val_idx = np.concatenate([np.where(groups == g)[0] for g in val_groups])
                    train_idx = np.setdiff1d(np.arange(total), val_idx)
                    if (len(np.unique(y.iloc[train_idx])) >= min_classes and
                        len(np.unique(y.iloc[val_idx])) >= min_classes):
                        return train_idx, val_idx

        return None, None

    train_local_idx, val_local_idx = split_train_val_by_subtype_strict(
        X_train_val, y_train_val, groups_train_val,
        min_ratio=VAL_RATIO_MIN,
        max_ratio=VAL_RATIO_MAX,
        min_classes=2,
        random_state=run + 200,
        max_attempts=500
    )

    if train_local_idx is None:
        print(f"⚠️ 无法在严格范围 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}] 内找到满足类别要求的验证集划分，跳过 Run {run+1}")
        skipped_runs += 1
        continue

    X_train = X_train_val.iloc[train_local_idx]
    y_train = y_train_val.iloc[train_local_idx]
    X_val = X_train_val.iloc[val_local_idx]
    y_val = y_train_val.iloc[val_local_idx]

    actual_val_ratio_trainval = len(X_val) / len(X_train_val)
    actual_val_ratio_total = len(X_val) / len(X)

    print(f"Train size: {len(X_train)} (Subtypes: {groups_train_val.iloc[train_local_idx].nunique()})")
    print(f"Val size: {len(X_val)} (Subtypes: {groups_train_val.iloc[val_local_idx].nunique()})")
    print(f"验证集占 train_val 比例: {actual_val_ratio_trainval:.4f} (严格范围 [{VAL_RATIO_MIN}, {VAL_RATIO_MAX}])")
    print(f"验证集占全体比例: {actual_val_ratio_total:.4f}")
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Val label distribution: {y_val.value_counts().to_dict()}")

    # =====================================================
    # 对每个模型训练（无交叉验证）
    # =====================================================
    for model_name, model in models.items():
        print("\n" + "#" * 100)
        print(f"Model: {model_name}")
        print("#" * 100)

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

        model.fit(X_tr, y_train)

        y_val_prob = model.predict_proba(X_vl)
        val_metrics = evaluate(y_val, y_val_prob)

        print("\nValidation Result:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        y_test_prob = model.predict_proba(X_te)
        test_metrics = evaluate(y_test, y_test_prob)


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
# 7. 汇总结果
# =========================================================
if len(all_results) == 0:
    print("\n" + "=" * 100)
    print("❌ 所有 Run 均因验证集划分失败而被跳过，无结果保存。")
    print("=" * 100)
else:
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 100)
    print("All Results")
    print("=" * 100)
    print(results_df)

    summary_mean = results_df.groupby("Model").mean()
    summary_std = results_df.groupby("Model").std()

    print("\nMean Results")
    print(summary_mean)
    print("\nStd Results")
    print(summary_std)

    results_df.to_csv("all_random_split_results_no_cv.csv", index=False)
    summary_mean.to_csv("summary_mean_no_cv.csv")
    summary_std.to_csv("summary_std_no_cv.csv")

    best_model = summary_mean["Val_AUROC"].idxmax()
    print("\n" + "=" * 100)
    print(f"Best Model Based On Validation Val_AUROC: {best_model}")
    print("=" * 100)

print(f"\n总 Run 数: {len(final_splits)}, 成功 {len(final_splits)-skipped_runs} 轮，跳过 {skipped_runs} 轮。")