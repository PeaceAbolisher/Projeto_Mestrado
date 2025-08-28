# === SHAP on your saved pipelines, with the same preprocessing they used ===
import os, joblib, shap, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- CONFIG -----------------
DATA_CSV = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\Obesity\all_data\merged_obesity_dataset.csv"

# >>> Here you can put full paths for each model <<<
MODEL_FILES = [
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\obesity_dataset\gans_augmented\obesity_best_model_gans_augmented.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\obesity_dataset\gans\obesity_best_model_gans.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\obesity_dataset\smote\obesity_best_model_smote.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\obesity_dataset\upsampling\obesity_best_model_upsampling.pkl",
]

OUTDIR = Path("obesity_shap_reports"); OUTDIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SUBSAMPLE = None        # ← use ALL test rows for explanations
BACKGROUND_SUBSAMPLE = None  # ← use ALL test rows as background (Kernel/Linear only)
POS_CLASS_INDEX = 1
# ------------------------------------------

# 1) Load and clean the original data exactly like it was trained
df = pd.read_csv(DATA_CSV, dtype=str)
if "sample_id" in df.columns:
    df = df.drop(columns=["sample_id"])
for c in df.columns:
    if c != "healthy":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X = df.drop(columns=["healthy"])
y = df["healthy"].astype(int)

#Train test split (validation does nothing here)
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

for pkl in MODEL_FILES:
    if not os.path.exists(pkl):
        print(f"[WARN] {pkl} not found, skipping.")
        continue

    print(f"\n[INFO] Explaining: {pkl}")
    pipe = joblib.load(pkl)
    model = pipe["model"]
    scaler = pipe["scaler"]
    feat_names = pipe["final_feature_names"]

    # --- PREPROCESSING ---
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    ).astype(np.float32)
    X_test_final = X_test_scaled.loc[:, feat_names]

    # EXPLAIN all the rows
    X_sample = X_test_final
    X_bg = X_test_final

    tag = Path(pkl).stem  # use the file name (without extension) for plots

    # 5) Choose explainer
    if is_tree(model):
        explainer = shap.TreeExplainer(model) #Drop feature_names= from the explainer constructors since X_sample has the right column names
        shap_values = explainer(X_sample)

        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - SHAP summary")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_summary.png", dpi=200); plt.close()

        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - mean(|SHAP|) bar")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_bar.png", dpi=200); plt.close()

    elif is_linear(model):
        explainer = shap.LinearExplainer(model, X_bg) #Drop feature_names= from the explainer constructors since X_sample has the right column names
        shap_values = explainer(X_sample)

        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - SHAP summary (linear)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_summary.png", dpi=200); plt.close()

        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - mean(|SHAP|) bar (linear)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_bar.png", dpi=200); plt.close()

    else:
        # SVM / KNN / MLP / Voting / Stacking → model-agnostic KernelExplainer
        def predict_pos(nd):
            df_ = pd.DataFrame(nd, columns=feat_names)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(df_)[:, POS_CLASS_INDEX]
            return model.decision_function(df_)

        explainer = shap.KernelExplainer(predict_pos, X_bg)
        shap_vals = explainer.shap_values(X_sample, nsamples="auto", l1_reg="aic")

        shap.summary_plot(shap_vals, X_sample, feature_names=feat_names, show=False, max_display=20)
        plt.title(f"{tag} - SHAP summary (Kernel)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_summary.png", dpi=200); plt.close()

        shap.summary_plot(shap_vals, X_sample, feature_names=feat_names, plot_type="bar", show=False, max_display=20)
        plt.title(f"{tag} - mean(|SHAP|) bar (Kernel)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_bar.png", dpi=200); plt.close()

    print(f"[INFO] Saved figures for {tag} to: {OUTDIR.resolve()}")
