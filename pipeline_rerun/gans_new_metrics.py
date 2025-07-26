# GANS Class Balancing
import numpy as np
import pandas as pd
import joblib
import copy
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.base import clone
from xgboost import XGBClassifier
import seaborn as sns
from itertools import combinations
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_mlp_model(input_dim=20, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_layers[0], activation='relu'))
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'AUC']) #logs acc and auc
    return model


print("\n[INFO] Starting GANS-Balanced Model Selection Pipeline...\n")
# --- Model Setup ---
models = {
    'rf': RandomForestClassifier(),
    'svm': SVC(probability=True), #enables -predict_proba() which is needed for soft voting or stacking classifiers with base estimators like a logistic regression, for example
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(),
    'gb': GradientBoostingClassifier(),
    'xgb': XGBClassifier()
}

param_grids = {
    'rf': { 'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30],'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],'bootstrap': [True, False] },
    'svm': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto'],  'degree': [3, 4, 5]},
    'knn': {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan'] },
    'lr': {'C': [0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2'],'solver': ['liblinear', 'saga'],'max_iter': [100, 500, 1000]},
    'gb': {'n_estimators': [100, 200],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 7],'subsample': [0.8, 1.0],'min_samples_split': [2, 5],'min_samples_leaf': [1, 2]},
    'xgb': {'n_estimators': [100, 200],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 7],'subsample': [0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.2]},
    'mlp': {'model__hidden_layers': [[64],[128, 64],[128, 64, 32]],'model__dropout_rate': [0.2, 0.3],'model__learning_rate': [0.001, 0.0005],'epochs': [30, 50],'batch_size': [16, 32]}
}



meta_learners = {
    'lr': LogisticRegression(max_iter=10000),
    'rf': RandomForestClassifier(),
    'gb': GradientBoostingClassifier()
}

scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler()
}


selection_methods = ['kbest', 'rfe']
best_estimators_final = {}

rfe_base_models = {
    'lr': LogisticRegression(max_iter=10000),
    'rf': RandomForestClassifier(),
    'svm': SVC(kernel='linear')
}
#Load dataset
data = pd.read_csv(r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\Project1\all_data_samples\merged_data.csv", dtype=str)

# Drop ID column (not useful for modeling)
data.drop(columns=["sample_id"], inplace=True)

# Convert all columns except 'Country' and 'healthy' to numeric
for col in data.columns:
    if col not in ['Country', 'healthy']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any missing values after conversion
data.dropna(inplace=True)

#This is needed since the dataset is ordered (first X r)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# --- Split data into training, validation, and test sets ---

# Separate features and target
X = data.drop(columns=['healthy'])
y = data['healthy'].astype(int)

# Step 1: Split off 20% for final test set  -- dev = 60% of training + 20% of validation
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 2: Split remaining 80% into training (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)


#--- Apply GAN (CTGAN) to balance the training set only ---

#balances on training data ONLY
train_df = X_train.copy()
train_df['healthy'] = y_train

# Identify class imbalance
minority_class = train_df['healthy'].value_counts().idxmin()
majority_class = train_df['healthy'].value_counts().idxmax()
samples_needed = train_df['healthy'].value_counts()[majority_class] - train_df['healthy'].value_counts()[minority_class]

# Generate synthetic samples only if needed
if samples_needed > 0:
    # Subset of minority class only
    minority_df = train_df[train_df['healthy'] == minority_class].copy()

    # Define metadata schema for CTGAN
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(minority_df)

    for column in minority_df.columns:
        if column not in ['Country', 'healthy', 'Age']:
            metadata.update_column(column_name=column, sdtype='numerical')
    metadata.update_column(column_name='Age', sdtype='numerical')
    metadata.update_column(column_name='Country', sdtype='categorical')

    # Fit CTGAN to minority data and sample to balance
    synthesizer = CTGANSynthesizer(metadata, epochs=300)
    synthesizer.fit(minority_df)
    synthetic_df = synthesizer.sample(samples_needed)

    # Combine real and synthetic into a balanced training set
    train_df = pd.concat([train_df, synthetic_df], ignore_index=True)

# Update training X and y with balanced data
X_train = train_df.drop(columns=['healthy'])
y_train = train_df['healthy']


# --- Encode and scale data using both scalers (standard, minmax) ---

# Fit encoder on training data
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train[['Country']])


best_performers = {}

for scaler_name, scaler in scalers.items():
    print(f"\n[INFO] Applying scaler: {scaler_name}")

    # --- Preprocess Training Set ---
    X_train_num = X_train.drop(columns=['Country'])
    X_train_cat = encoder.transform(X_train[['Country']]).toarray()

    # Fit scaler on training numeric data
    scaler.fit(X_train_num)
    X_train_scaled = scaler.transform(X_train_num)

    # Combine scaled numeric + encoded categorical
    X_train_processed = pd.concat([
        pd.DataFrame(X_train_scaled, columns=X_train_num.columns),
        pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(['Country']))
    ], axis=1).astype(np.float32)

    # --- Preprocess Validation Set ---
    X_val_num = X_val.drop(columns=['Country'])
    X_val_cat = encoder.transform(X_val[['Country']]).toarray()
    X_val_scaled = scaler.transform(X_val_num)

    X_val_processed = pd.concat([
        pd.DataFrame(X_val_scaled, columns=X_val_num.columns),
        pd.DataFrame(X_val_cat, columns=encoder.get_feature_names_out(['Country']))
    ], axis=1).astype(np.float32)

    # --- Estimate top-k features using RFECV on training data ---

    print("\nFeature Selection: Computing optimal k-values using RFECV")

    rfe_k_values = {}  # Stores best number of features per base model
    rfe_models = {}
    for base_name, base_model in rfe_base_models.items():
        # Use RFECV to find optimal number of features (based on CV AUC)
        rfecv = RFECV(
            estimator=base_model,
            step=1,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        rfecv.fit(X_train_processed, y_train)
        #Take the smallest number between 20 and the number of features the RFECV found
        #with our GOAL - classify the microbiome. if K (number of features) is too high there is no way we can classify the microbiome
        k = min(20, rfecv.n_features_)
        rfe_k_values[base_name] = k
        rfe_models[base_name] = rfecv  # Save full model to reuse it later instead of using a new one to actually get the features

        print(f"RFECV ({base_name.upper()}) selected top {k} features (capped at 20)")


    # --- Run both RFE and KBest with selected k per base model ---

    for method in selection_methods:
        for base_name, base_model in rfe_base_models.items():
            k = rfe_k_values[base_name]
            print(f"\n[INFO] Feature Selection: {method.upper()} | Base: {base_name.upper()} | k = {k}")

            if method == 'rfe':
              selector = rfe_models[base_name] #RFECV object that was saved
              ranking = selector.ranking_  # RFE.ranking_: 1 = most important; we sort ascending
              feature_ranks = pd.Series(ranking, index=X_train_processed.columns)
              top_features = feature_ranks.sort_values().index[:k]
              selected_features = top_features
              selector_obj = None  # RFE doesn't have a .transform() method, so we don't reuse it later.
            # We just keep the feature names. (Unlike KBest, which is reused during test-time.)

            elif method == 'kbest':
              selector = SelectKBest(score_func=mutual_info_classif, k=k)
              selector.fit(X_train_processed, y_train)
              scores = selector.scores_ #KBest.scores_: Higher = more relevant; we sort descending
              feature_scores = pd.Series(scores, index=X_train_processed.columns)
              top_features = feature_scores.sort_values(ascending=False).index[:k]
              selected_features = top_features
              selector_obj = selector  # Needed for test-time inference

            # --- Filter training data to include only selected features ---
            # These are the features selected by either RFE or KBest (depending on the current loop)
            X_train_selected = X_train_processed[selected_features]
            X_val_selected = X_val_processed[selected_features]

            # --- Add MLP model now that input size is known ---
            models['mlp'] = KerasClassifier(
                model=create_mlp_model,
                model__input_dim=X_train_selected.shape[1],
                verbose=0
            )

            # --- Train base models with GridSearchCV and store best estimators ---
            best_estimators = {}
            single_results = {}
            for name, model in models.items():
                print(f"Training model: {name.upper()} with GridSearchCV")

                grid_params = copy.deepcopy(param_grids[name]) 
                #Fix input size mismatch for MLP -->prevents missmatch
                if name == 'mlp':
                    grid_params['model__input_dim'] = [X_train_selected.shape[1]] #only changing the local grid for that run, instead of changing it globally

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=grid_params, #uses the deep_copied params
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid.fit(X_train_selected, y_train)

                #evaluates base models on validation set
                y_val_pred = grid.predict(X_val_selected)
                val_auc = None
                val_precision = precision_score(y_val, y_val_pred)
                val_recall = recall_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred)
                if hasattr(grid, "predict_proba"):
                    y_val_proba = grid.predict_proba(X_val_selected)[:, 1]
                    try:
                        val_auc = roc_auc_score(y_val, y_val_proba)
                    except ValueError:
                        val_auc = None


                label = f"Single-{name}"
                single_results[label] = {
                    'auc': val_auc,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1,
                    'model': grid.best_estimator_
                }


                best_estimators[name] = grid.best_estimator_
                print(
                    f"Best {name.upper()} | "
                    f"Recall: {val_recall:.3f} | "
                    f"Precision: {val_precision:.3f} | "
                    f"F1: {val_f1:.3f} | "
                    f"AUC: {f'{val_auc:.3f}' if val_auc is not None else 'N/A'}"
                )


           # --- Build Voting and Stacking Ensembles using all base models ---
            model_names = [name for name in best_estimators.keys() if name != 'mlp'] #leave MLP outside of ensemble due to small dataset and computational strain... not worth it

            voting_results = {}
            stacking_results = {}

            # Try all combinations of 2 or more models
            for i in range(2, len(model_names) + 1):
                for combo in combinations(model_names, i):
                    estimators = [(name, best_estimators[name]) for name in combo]

                    # --- Try both hard and soft voting ---
                    for voting_type in ['hard', 'soft']:
                        try:
                            voting_model = VotingClassifier(
                                estimators=estimators,
                                voting=voting_type
                            )
                            voting_model.fit(X_train_selected, y_train)
                            y_pred = voting_model.predict(X_val_selected)
                            val_auc = None
                            if hasattr(voting_model, "predict_proba"):
                                y_val_proba = voting_model.predict_proba(X_val_selected)[:, 1]
                                try:
                                    val_auc = roc_auc_score(y_val, y_val_proba)
                                except ValueError:
                                    val_auc = None


                            label = f"Voting-{voting_type}-{combo}"
                            voting_results[label] = {
                                'auc': val_auc,
                                'precision': precision_score(y_val, y_pred),
                                'recall': recall_score(y_val, y_pred),
                                'f1': f1_score(y_val, y_pred),
                                'model': voting_model
                            }
                            print(
                                f"{label} | "
                                f"Recall: {recall_score(y_val, y_pred):.3f} | "
                                f"Precision: {precision_score(y_val, y_pred):.3f} | "
                                f"F1: {f1_score(y_val, y_pred):.3f} | "
                                f"AUC: {f'{val_auc:.3f}' if val_auc is not None else 'N/A'} |"
                            )

                        except Exception:
                            continue  # Some combos may fail (e.g., soft voting needs predict_proba)

                    # --- Try stacking with each meta-learner ---
                    for meta_name, meta_model in meta_learners.items():
                        try:
                          stacking_model = StackingClassifier(
                              estimators=estimators,
                              final_estimator=clone(meta_model),
                              passthrough=True,
                              cv=3,
                              n_jobs=-1
                          )
                          stacking_model.fit(X_train_selected, y_train)
                          y_pred = stacking_model.predict(X_val_selected)
                          val_auc = None
                          if hasattr(stacking_model, "predict_proba"):
                            y_val_proba =  stacking_model.predict_proba(X_val_selected)[:, 1]
                            try:
                                val_auc = roc_auc_score(y_val, y_val_proba) 
                            except ValueError:
                                val_auc = None


                          label = f"Stacking-{meta_name}-{combo}"
                          stacking_results[label] = {
                            'auc': val_auc,
                            'precision': precision_score(y_val, y_pred),
                            'recall': recall_score(y_val, y_pred),
                            'f1': f1_score(y_val, y_pred),
                            'model': stacking_model

                        }
                          print(
                            f"{label} | "
                            f"Recall: {recall_score(y_val, y_pred):.3f} | "
                            f"Precision: {precision_score(y_val, y_pred):.3f} | "
                            f"F1: {f1_score(y_val, y_pred):.3f} | "
                           f"AUC: {f'{val_auc:.3f}' if val_auc is not None else 'N/A'} |"
                        )
                        except Exception:
                            continue

            # --- Track the best performing ensemble for this configuration ---
            all_results = {**single_results, **voting_results, **stacking_results}
            if all_results:

                def safe_recall_key(item):
                    metrics = all_results[item]
                    recall = metrics.get('recall', -1)
                    precision = metrics.get('precision', -1)
                    f1 = metrics.get('f1', -1)
                    return (recall, precision, f1)
                best_label = max(all_results, key=safe_recall_key)

                result_key = f"ctgan-{scaler_name}-{method}-{base_name}-{best_label}"
                val_auc = all_results[best_label].get('auc', -1)

                print(f"Candidate best: {result_key} | "f"Recall: {all_results[best_label]['recall']:.3f} | "f"Precision: {all_results[best_label]['precision']:.3f} | "f"F1: {all_results[best_label]['f1']:.3f}" )

                # Deep copy model and selector for best config
                selector_copy = selector_obj if method == 'kbest' and selector_obj is not None else None

                model_copy = all_results[best_label]['model']

                # Prepare partial pipeline (test_eval done later)
                best_performers[result_key] = {
                    'pipeline': {
                        'model': model_copy,
                        'scaler': copy.deepcopy(scaler),
                        'selector': selector_copy,
                        'encoder': copy.deepcopy(encoder),
                        'feature_method': method,
                        'base_model': base_name,
                        'k': k,
                        'label': best_label,
                        'pre_selection_feature_names': list(X_train_processed.columns),
                        'final_feature_names': list(selected_features),
                        'scaler_type': scaler_name,
                        'balancing_method': 'gan_ctgan',
                        'val_auc': val_auc,
                        'val_precision': all_results[best_label]['precision'],
                        'val_recall': all_results[best_label]['recall'],
                        'val_f1': all_results[best_label]['f1'],
                        'augmentation': None
                    },
                }

#select based on validation recall, precision and f1_score
sorted_models = sorted(
    best_performers.items(),
    key=lambda x: (
        -x[1]['pipeline'].get('val_recall', -1),
        -x[1]['pipeline'].get('val_precision', -1),
        -x[1]['pipeline'].get('val_f1', -1)
    )
)

best_model_name, best_entry = sorted_models[0]  # selecting best model based on recall → precision → F1
best_model_pipeline = best_entry['pipeline']

# === Evaluate Final Model on Test Set ===
model = best_model_pipeline['model']
scaler = best_model_pipeline['scaler']
encoder = best_model_pipeline['encoder']
selector = best_model_pipeline['selector']
features = best_model_pipeline['final_feature_names']

# Preprocess test set
X_test_num = X_test.drop(columns=['Country'])
X_test_scaled = scaler.transform(X_test_num)
X_test_cat = encoder.transform(X_test[['Country']]).toarray()
X_test_proc = pd.concat([
    pd.DataFrame(X_test_scaled, columns=X_test_num.columns),
    pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(['Country']))
], axis=1).astype(np.float32)

# Feature selection for test set
if selector is not None:
    X_test_final = X_test_proc.loc[:, selector.get_support()] #Slelect the actual columns kept by the selector, in the right order
else:
    X_test_final = X_test_proc[features]

# Predict
y_test_pred = model.predict(X_test_final)
true_test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = None
y_proba = None
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test_final)[:, 1]
    try:
        test_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        test_auc = None


test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
pr_auc = average_precision_score(y_test, y_proba) if test_auc is not None else None

print("\n[INFO] FINAL MODEL SUMMARY")
print("=" * 40)
print(f"[INFO] Best Config        : {best_model_name}")
print(f"Validation Recall   : {best_model_pipeline['val_recall']:.3f}")
print(f"Validation Precision: {best_model_pipeline['val_precision']:.3f}")
print(f"Validation F1-Score : {best_model_pipeline['val_f1']:.3f}")
print(f"Test Accuracy       : {true_test_accuracy:.3f}")
print(f"Test ROC-AUC        : {test_auc:.3f}" if test_auc is not None else "")
print(f"Test PR-AUC         : {pr_auc:.3f}" if pr_auc is not None else "")
print(f"Test Recall         : {test_recall:.3f}")
print(f"Test Precision      : {test_precision:.3f}")
print(f"Test F1-Score       : {test_f1:.3f}")
print("=" * 40)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


if y_proba is not None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()



# Save test metrics
best_model_pipeline['test_accuracy'] = true_test_accuracy
best_model_pipeline['test_auc'] = test_auc
best_model_pipeline['test_precision'] = test_precision
best_model_pipeline['test_recall'] = test_recall
best_model_pipeline['test_f1'] = test_f1
best_model_pipeline['test_pr_auc'] = pr_auc
joblib.dump(best_model_pipeline, 'best_model_gans_new_metrics.pkl')