import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

negative_sequences_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/negative_protein_sequences.csv"
positive_sequences_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/positive_protein_sequences.csv"

negative_sequences_df = pd.read_csv(negative_sequences_path, encoding='ISO-8859-1')
positive_sequences_df = pd.read_csv(positive_sequences_path, encoding='ISO-8859-1')

positive_sequences_df["PPI"] = 1
negative_sequences_df["PPI"] = 0
df = pd.concat([positive_sequences_df, negative_sequences_df], axis=0).reset_index(drop=True)

amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

df["protein1_length"] = df["protein_sequences_1"].str.len()
df["protein2_length"] = df["protein_sequences_2"].str.len()

def calculate_properties(seq):
    analysis = ProteinAnalysis(seq)
    return {
        'Molecular_Weight': analysis.molecular_weight(),
        'Aromaticity': analysis.aromaticity(),
        'Isoelectric_Point': analysis.isoelectric_point(),
        'Helix': analysis.secondary_structure_fraction()[0],
        'Turn': analysis.secondary_structure_fraction()[1],
        'Sheet': analysis.secondary_structure_fraction()[2]
    }

df = pd.concat([df, df['protein_sequences_1'].apply(calculate_properties).apply(pd.Series).add_prefix('p1_')], axis=1)
df = pd.concat([df, df['protein_sequences_2'].apply(calculate_properties).apply(pd.Series).add_prefix('p2_')], axis=1)

feature_cols = [col for col in df.columns if col not in ['protein_sequences_1', 'protein_sequences_2', 'PPI']]
X = df[feature_cols].values
y = df['PPI'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grids = {
    "XGBoost": {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    },
    "Random Forest": {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
}

models = {
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_models = {}
train_accuracies = {}
test_accuracies = {}

for model_name, model in models.items():
    print(f"Tuning {model_name}...")

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[model_name],
        n_iter=10,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_models[model_name] = random_search.best_estimator_

    y_train_pred = best_models[model_name].predict(X_train)
    y_test_pred = best_models[model_name].predict(X_test)

    train_accuracies[model_name] = accuracy_score(y_train, y_train_pred)
    test_accuracies[model_name] = accuracy_score(y_test, y_test_pred)

print("\nFinal Model Accuracies:\n")
for model_name in best_models.keys():
    print(f"{model_name}:")
    print(f"  Train Accuracy: {train_accuracies[model_name]:.4f}")
    print(f"  Test Accuracy:  {test_accuracies[model_name]:.4f}\n")
