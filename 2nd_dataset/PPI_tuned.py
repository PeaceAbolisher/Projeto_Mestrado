import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# File paths
negative_sequences_path = "C:/Users/Rafael Fonseca/Downloads/negative_protein_sequences.csv"
positive_sequences_path = "C:/Users/Rafael Fonseca/Downloads/positive_protein_sequences.csv"

# Load data
negative_sequences_df = pd.read_csv(negative_sequences_path, encoding='ISO-8859-1')
positive_sequences_df = pd.read_csv(positive_sequences_path, encoding='ISO-8859-1')

# Label data and merge
positive_sequences_df["PPI"] = 1
negative_sequences_df["PPI"] = 0
df = pd.concat([positive_sequences_df, negative_sequences_df], axis=0).reset_index(drop=True)

# Standard amino acids list
amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Calculate sequence lengths
df["protein1_length"] = df["protein_sequences_1"].str.len()
df["protein2_length"] = df["protein_sequences_2"].str.len()

# Function to calculate protein properties
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

# Calculate properties for both protein sequences
df = pd.concat([df, df['protein_sequences_1'].apply(calculate_properties).apply(pd.Series).add_prefix('p1_')], axis=1)
df = pd.concat([df, df['protein_sequences_2'].apply(calculate_properties).apply(pd.Series).add_prefix('p2_')], axis=1)

# Get feature columns
feature_cols = [col for col in df.columns if col not in ['protein_sequences_1', 'protein_sequences_2', 'PPI']]
X = df[feature_cols].values
y = df['PPI'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grids for RandomizedSearchCV
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

# Models to optimize
models = {
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Dictionary to store the best models after tuning
best_models = {}

# Perform hyperparameter tuning with RandomizedSearchCV
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    
    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_grids[model_name], 
        n_iter=50,  # Adjust based on desired speed/accuracy tradeoff
        scoring='accuracy', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Fit the model to find the best parameters
    random_search.fit(X_train, y_train)
    
    # Store the best model
    best_models[model_name] = random_search.best_estimator_
    
    # Evaluate and print the accuracy on the test set
    y_test_pred = best_models[model_name].predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{model_name} Best Test Accuracy after Tuning: {test_accuracy:.2f}")
    print(f"Best Parameters: {random_search.best_params_}\n")

# Display results in a confusion matrix
#for model_name, model in best_models.items():
#    y_test_pred = model.predict(X_test)
 #   cm = confusion_matrix(y_test, y_test_pred)
 #   plt.figure(figsize=(6, 5))
#    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#    plt.xlabel('Predicted')
#    plt.ylabel('Actual')
#    plt.title(f'Confusion Matrix - {model_name}')
#    plt.show()

# Display feature importances for models that support it
#for model_name, model in best_models.items():
 #   if hasattr(model, "feature_importances_"):
  #      plt.figure(figsize=(10, 8))
   #     importances = model.feature_importances_
    #    indices = np.argsort(importances)[::-1]
     #   top_features = 10  # Number of top features to display
#
 #       plt.bar(range(top_features), importances[indices[:top_features]], align='center')
  #      plt.xticks(range(top_features), np.array(feature_cols)[indices[:top_features]], rotation=90)
   #     plt.title(f'Top {top_features} Feature Importances - {model_name}')
    #    plt.show()
