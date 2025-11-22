import os, joblib, shap, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- CONFIG -----------------
DATA_CSV = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\Hypertension\multi-label\all_data_samples\merged_hypertensive_dataset_multiclass.csv"

MODEL_FILES = [
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\hypertension_multi_label\gans_augmented\hypertension_multi_class_best_model_gans_augmented.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\hypertension_multi_label\gans\hypertension_multi_class_best_model_gans.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\hypertension_multi_label\smote\hypertension_multi_class_best_model_smote.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\hypertension_multi_label\upsampling\hypertension_multi_class_best_model_upsampling.pkl",
]

OUTDIR = Path("hypertension_shap_reports"); OUTDIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
# ------------------------------------------

# 1) Load and clean data EXACTLY like for multiclass training
df = pd.read_csv(DATA_CSV, dtype=str)

# Drop ID column if present
if "sample_id" in df.columns:
    df = df.drop(columns=["sample_id"])

# Convert all non-target columns to numeric
for c in df.columns:
    if c != "bp_class":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with NaNs and shuffle (same as training scripts)
df = df.dropna().sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Features / multiclass target
X = df.drop(columns=["bp_class"])
y = df["bp_class"].astype(int)


# Same test split used in training
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

def is_tree(m):
    if isinstance(m, (RandomForestClassifier, GradientBoostingClassifier)):
        return True
    if HAS_XGB and isinstance(m, XGBClassifier):
        return True
    return False

def is_linear(m):
    return isinstance(m, LogisticRegression)

# ------------ MAIN LOOP ------------
for pkl in MODEL_FILES:
    if not os.path.exists(pkl):
        print(f"[WARN] {pkl} not found, skipping.")
        continue

    print(f"\n[INFO] Explaining MULTICLASS model: {pkl}")
    pipe = joblib.load(pkl)
    model = pipe["model"]
    scaler = pipe["scaler"]
    feat_names = pipe["final_feature_names"]

    # Preprocess
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    ).astype(np.float32)

    X_test_final = X_test_scaled.loc[:, feat_names]

    tag = Path(pkl).stem

    # Number of classes
    n_classes = len(model.classes_)

    # -------- TREE MODELS ---------
    if is_tree(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_final)

        # shap_values is list: one array per class
        for c in range(n_classes):
            shap.plots.beeswarm(shap_values[c], max_display=20, show=False)
            plt.title(f"{tag} - SHAP summary (class {c})")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{tag}_summary_class{c}.png", dpi=200)
            plt.close()

            shap.plots.bar(shap_values[c], max_display=20, show=False)
            plt.title(f"{tag} - mean(|SHAP|) class {c}")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{tag}_bar_class{c}.png", dpi=200)
            plt.close()

    # -------- LINEAR MODELS ---------
    elif is_linear(model):
        explainer = shap.LinearExplainer(model, X_test_final)
        shap_values = explainer.shap_values(X_test_final)

        for c in range(n_classes):
            shap.summary_plot(shap_values[c], X_test_final, show=False, max_display=20)
            plt.title(f"{tag} - SHAP summary (linear) class {c}")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{tag}_summary_class{c}.png", dpi=200)
            plt.close()

    # -------- KERNEL EXPLAINER (SVM, MLP, Voting, Stacking, etc.) ---------
    else:
        def predict_multiclass(nd):
            d = pd.DataFrame(nd, columns=feat_names)
            return model.predict_proba(d)  # shape (N, n_classes)

        # background = test set (you can subsample if you want speed)
        X_bg = X_test_final

        explainer = shap.KernelExplainer(predict_multiclass, X_bg)
        shap_values = explainer.shap_values(X_test_final)

        for c in range(n_classes):
            # IMPORTANT: do NOT pass X_test_final here, only shap_values + feature_names
            shap.summary_plot(
                shap_values[c],
                feature_names=feat_names,
                max_display=20,
                show=False
            )
            plt.title(f"{tag} - SHAP summary (Kernel) class {c}")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{tag}_summary_class{c}.png", dpi=200)
            plt.close()


    print(f"[INFO] Saved SHAP plots for {tag} into {OUTDIR}")
