# === SHAP on your saved pipelines, with the same preprocessing they used ===
import os, joblib, shap, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier  # needed for unpickling
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#allow unpickling models that reference create_mlp_model / KerasClassifier
def create_mlp_model(input_dim=20, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_layers[0], activation='relu')); model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu')); model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=[])
    return model

# compatibility shims for pickles saved with different module paths
import sys, types, __main__
shim = types.ModuleType("my_models")
shim.create_mlp_model = create_mlp_model
sys.modules["my_models"] = shim                 # supports my_models.create_mlp_model
__main__.create_mlp_model = create_mlp_model    # supports __main__.create_mlp_model


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- CONFIG -----------------
DATA_CSV = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\diabetesv2\all_data_samples\project2_merged_data.csv"

# >>> Here you can put full paths for each model <<<
MODEL_FILES = [
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2_sem_metadata\gans_augmented\best_model_gans_augmented_v2_sem_metadata.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2_sem_metadata\gans\best_model_gans_v2_sem_metadata.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2_sem_metadata\smote\best_model_smote_v2_sem_metadata.pkl",
    r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\output\t2d_dataset_v2_sem_metadata\upsampling\best_model_upsampling_v2_sem_metadata.pkl",
]
OUTDIR = Path("t2d_v2shap_reports_sem_metadata"); OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SUBSAMPLE = None        # ← use ALL test rows for explanations
BACKGROUND_SUBSAMPLE = None  # ← use ALL test rows as background (Kernel/Linear only)
POS_CLASS_INDEX = 1
# ------------------------------------------

data = pd.read_csv(DATA_CSV, dtype=str)

# Drop ID column (not useful for modeling), BMI (58% missing), Country (constant = China), and demographic metadata (Age, Gender)
data.drop(columns=["sample_id", "BMI", "Country", "Age", "Gender"], inplace=True)

# Convert all columns except 'healthy' to numeric            -----> Country is always China so it's irrelevant for learning
for col in data.columns:
    if col not in ['healthy']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any missing values after conversion
data.dropna(inplace=True)

#This is needed since the dataset is ordered (first X r)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

if len(data) == 0:
    raise ValueError("No rows left after cleaning — check column types and NaNs.")

X = data.drop(columns=['healthy'])
y = data['healthy'].astype(int)

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

    # --- PREPROCESSING (no metadata, no encoder) ---
    # Be safe and drop metadata again if present
    drop_cols = [c for c in ['sample_id', 'Country', 'Age'] if c in X_test.columns]
    X_num = X_test.drop(columns=drop_cols)

    # Scale numeric features with the saved scaler (fitted during training)
    X_scaled = pd.DataFrame(scaler.transform(X_num), columns=X_num.columns)

    # If a pre-selection order was saved, enforce it before final selection
    if pipe.get('pre_selection_feature_names'):
        X_scaled = X_scaled.loc[:, pipe['pre_selection_feature_names']]

    # Select the exact training features (names + order)
    feat_names = pipe['final_feature_names']  # refresh from pipe, not earlier var
    X_test_final = X_scaled.loc[:, feat_names].astype(np.float32)

    # Use all rows for SHAP by default
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
        # --- PATCH 2: robust positive-class scorer for KernelExplainer
        def predict_pos(nd):
            df_ = pd.DataFrame(nd, columns=feat_names)

            # Prefer probabilities if available
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(df_))

                # (n,) already positive-class probs
                if proba.ndim == 1:
                    return proba

                # (n,1) squeeze to (n,)
                if proba.ndim == 2 and proba.shape[1] == 1:
                    return proba[:, 0]

                # (n,2) or more → choose column for class "1" when known
                pos_idx = None
                if hasattr(model, "classes_"):
                    try:
                        pos_idx = int(np.where(model.classes_ == 1)[0][0])
                    except Exception:
                        pos_idx = None
                if pos_idx is None:
                    pos_idx = POS_CLASS_INDEX  # fallback
                return proba[:, pos_idx]

            # Fall back to decision_function → map to (0,1) with sigmoid
            if hasattr(model, "decision_function"):
                margin = np.asarray(model.decision_function(df_))
                if margin.ndim == 1:
                    return 1.0 / (1.0 + np.exp(-margin))
                pos_idx = None
                if hasattr(model, "classes_"):
                    try:
                        pos_idx = int(np.where(model.classes_ == 1)[0][0])
                    except Exception:
                        pos_idx = None
                if pos_idx is None:
                    pos_idx = POS_CLASS_INDEX
                return 1.0 / (1.0 + np.exp(-margin[:, pos_idx]))

            # Last resort: use predicted labels as float
            return np.asarray(model.predict(df_), dtype=float)
        # --- END PATCH 2


        explainer = shap.KernelExplainer(predict_pos, X_bg)
        shap_vals = explainer.shap_values(X_sample, nsamples="auto", l1_reg="aic")

        shap.summary_plot(shap_vals, X_sample, feature_names=feat_names, show=False, max_display=20)
        plt.title(f"{tag} - SHAP summary (Kernel)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_summary.png", dpi=200); plt.close()

        shap.summary_plot(shap_vals, X_sample, feature_names=feat_names, plot_type="bar", show=False, max_display=20)
        plt.title(f"{tag} - mean(|SHAP|) bar (Kernel)")
        plt.tight_layout(); plt.savefig(OUTDIR / f"{tag}_bar.png", dpi=200); plt.close()

    print(f"[INFO] Saved figures for {tag} to: {OUTDIR.resolve()}")
