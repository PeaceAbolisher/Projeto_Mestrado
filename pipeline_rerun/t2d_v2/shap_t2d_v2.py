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

#Some tree classifiers (and certain SHAP versions) return a single-output Explanation for binary tasks, while others return both class outputs.
# Your second model produced the latter ([..., 2]), so beeswarm complained. Slicing to the positive class (index 1) fixes it consistently.
def pick_output(expl, pos_index=1):
    """Return a 2D Explanation for the selected class.
       Works with both new (Explanation) and old (list-per-class) SHAP APIs."""
    # Newer SHAP: Explanation object
    if hasattr(expl, "values"):
        vals = expl.values
        if isinstance(vals, np.ndarray) and vals.ndim == 3:
            # keep the Explanation type by slicing along the output axis
            return expl[..., pos_index]
        return expl
    # Older SHAP: list of arrays/Explanations per class
    if isinstance(expl, (list, tuple)):
        return expl[pos_index]
    return expl


# ----------------- CONFIG -----------------
DATA_CSV = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\diabetesv2\all_data_samples\project2_merged_data.csv"

# >>> Here you can put full paths for each model <<<
MODEL_FILES = [
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2\gans_augmented\best_model_gans_augmented_v2.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2\gans\best_model_gans_v2.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2\smote\best_model_smote_v2.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2\upsampling\best_model_upsampling_v2.pkl",
]

OUTDIR = Path("t2d_v2_shap_reports"); OUTDIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SUBSAMPLE = None        # ← use ALL test rows for explanations
BACKGROUND_SUBSAMPLE = None  # ← use ALL test rows as background (Kernel/Linear only)
POS_CLASS_INDEX = 1
# ------------------------------------------

df = pd.read_csv(DATA_CSV, dtype=str)

# EXACTLY match training drops
drop_cols = [c for c in ("sample_id", "BMI", "Country") if c in df.columns]
df = df.drop(columns=drop_cols)

# Gender mapping (same as training)
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].str.strip().str.lower().map({"male": 0, "female": 1})

# Convert all non-target columns to numeric (same as training)
for c in df.columns:
    if c != "healthy":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with any missing values, then shuffle (same as training)
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

    # --- PREPROCESSING (match training-v2 exactly) ---
    scaler     = pipe["scaler"]
    feat_names = pipe["final_feature_names"]

    # Use the same column order the scaler saw during training, if saved
    if pipe.get("pre_selection_feature_names"):
        precols = pipe["pre_selection_feature_names"]
        # safety: make sure sets match
        if set(precols) != set(X_test.columns):
            missing_in_test = [c for c in precols if c not in X_test.columns]
            extra_in_test   = [c for c in X_test.columns if c not in precols]
            raise ValueError(f"Column mismatch before scaling.\nMissing in test: {missing_in_test}\nExtra in test: {extra_in_test}")
        X_to_scale = X_test.loc[:, precols]
    else:
        X_to_scale = X_test

    #scale numeric features only (all remaining are numeric in v2)
    X_all = pd.DataFrame(scaler.transform(X_to_scale), columns=X_to_scale.columns)

    missing = [c for c in feat_names if c not in X_all.columns]
    if missing:
        raise ValueError(f"Missing features in SHAP input: {missing}")

    #exact training feature matrix (names + order)
    X_test_final = X_all.loc[:, feat_names].astype(np.float32)

    #rows to explain / background
    X_sample = X_test_final
    X_bg = (X_test_final if BACKGROUND_SUBSAMPLE is None
            else X_test_final.sample(BACKGROUND_SUBSAMPLE, random_state=RANDOM_STATE))


    tag = Path(pkl).stem  # use the file name (without extension) for plots

    # --- Tree models ---
#TreeExplainer has two main modes for trees:
    #tree_path_dependent (the legacy/default in many cases): it walks the training tree paths to estimate missing-feature expectations. 
        #In this mode, SHAP only supports the model’s raw margin (e.g., log-odds for logistic trees), not probabilities.

    #interventional: it replaces “missing” features by sampling from a background dataset you provide. 
        # This mode does support model_output="probability".

#Your second model ended up in tree_path_dependent mode internally (SHAP auto-picks based on model/details). 
# Because that mode doesn’t allow model_output="probability", asking for probabilities triggered the error. By explicitly selecting:feature_perturbation="interventional" andpassing a background dataset (data=X_bg) thus fixing the error
    if is_tree(model): 
        try:
            explainer = shap.TreeExplainer(
                model,
                data=X_bg,  # background for interventional
                feature_perturbation="interventional",
                model_output="probability",
            )
            shap_values = explainer(X_sample)
        except ValueError:
            # Fallback if a given model still refuses probability
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer(X_sample)  # raw log-odds / margin

        shap_values = pick_output(shap_values, POS_CLASS_INDEX)

        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - SHAP summary")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_summary.png", dpi=200); plt.close()

        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.title(f"{tag} - mean(|SHAP|) bar")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_bar.png", dpi=200); plt.close()

    # --- Linear models ---
    elif is_linear(model):
        explainer = shap.LinearExplainer(model, X_bg)
        shap_values = explainer(X_sample)
        shap_values = pick_output(shap_values, POS_CLASS_INDEX)  # <— add this

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
