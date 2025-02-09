import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

negative_sequences_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/negative_protein_sequences.csv"
positive_sequences_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/positive_protein_sequences.csv"

# Read the CSV files instead of Excel
negative_sequences_df = pd.read_csv(negative_sequences_path)
positive_sequences_df = pd.read_csv(positive_sequences_path)

# Label the data and concatenate
positive_sequences_df["PPI"] = 1
negative_sequences_df["PPI"] = 0
df = pd.concat([positive_sequences_df, negative_sequences_df], axis=0).reset_index(drop=True)

# Define standard amino acids
amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# (Optional) Function to check for non-standard amino acids
def count_non_standard(sequence):
    return bool(set(sequence) - amino_acids)

# (Optional) Count sequences with non-standard amino acids
# non_standard_count_1 = df['protein_sequences_1'].apply(count_non_standard).sum()
# non_standard_count_2 = df['protein_sequences_2'].apply(count_non_standard).sum()
# print("Non-standard amino acids count in protein 1:", non_standard_count_1)
# print("Non-standard amino acids count in protein 2:", non_standard_count_2)

# Calculate sequence lengths
df["protein1_length"] = df["protein_sequences_1"].str.len()
df["protein2_length"] = df["protein_sequences_2"].str.len()

# Calculate amino acid composition for both sequences
def aa_comp(seq):
    return {aa: round(seq.count(aa) / len(seq) * 100, 3) for aa in amino_acids}

df = pd.concat([df, df['protein_sequences_1'].apply(aa_comp).apply(pd.Series).add_prefix('p1_')], axis=1)
df = pd.concat([df, df['protein_sequences_2'].apply(aa_comp).apply(pd.Series).add_prefix('p2_')], axis=1)

# Calculate additional protein properties
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

# Select feature columns (exclude raw sequences and target label)
feature_cols = [col for col in df.columns if col not in ['protein_sequences_1', 'protein_sequences_2', 'PPI']]

# (Optional) Plot boxplots for each feature by PPI group
# for col in feature_cols:
#     sns.boxplot(x='PPI', y=col, data=df)
#     plt.xlabel('PPI Occurrence')
#     plt.ylabel(col)
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# Prepare the feature matrix and target vector
X = df[feature_cols].values
y = df['PPI'].values

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

# Define models for evaluation
models = {
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Neural Network": MLPClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

train_accuracies = {}
test_accuracies = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_accuracies[model_name] = train_acc
    test_accuracies[model_name] = test_acc

print("Model Accuracies:\n")
for model_name in models.keys():
    print(f"{model_name}:")
    print(f"  Train Accuracy: {train_accuracies[model_name]:.2f}")
    print(f"  Test Accuracy:  {test_accuracies[model_name]:.2f}\n")
