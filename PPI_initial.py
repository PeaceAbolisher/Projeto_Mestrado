import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

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

# Function to check for non-standard amino acids
def count_non_standard(sequence):
    return bool(set(sequence) - amino_acids)

# Count sequences with non-standard amino acids
    #count_non_standard1 = sum(df['protein_sequences_1'].apply(count_non_standard))
    #count_non_standard2 = sum(df['protein_sequences_2'].apply(count_non_standard))

# Calculate sequence lengths
df["protein1_length"] = df["protein_sequences_1"].str.len()
df["protein2_length"] = df["protein_sequences_2"].str.len()

# Function to calculate amino acid composition as percentage
def aa_comp(seq):
    return {aa: round(seq.count(aa) / len(seq) * 100, 3) for aa in amino_acids}

# Calculate amino acid composition for both sequences
    #df = pd.concat([df, df['protein_sequences_1'].apply(aa_comp).apply(pd.Series).add_prefix('p1_')], axis=1)
    #df = pd.concat([df, df['protein_sequences_2'].apply(aa_comp).apply(pd.Series).add_prefix('p2_')], axis=1)

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

# Get feature columns for box plotting
feature_cols = [col for col in df.columns if col not in ['protein_sequences_1', 'protein_sequences_2', 'PPI']]

# Plot each feature in a boxplot by PPI group
for col in feature_cols:
    sns.boxplot(x='PPI', y=col, data=df)
    plt.xlabel('PPI Occurrence')
    plt.ylabel(col)

# Assuming 'X' contains your feature matrix and 'y' contains the target labels
X = df[feature_cols].values
y = df['PPI'].values

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

# Define your models again if necessary, or reuse the models dictionary
models = {
    #"Support Vector Classifier": SVC(probability=True, random_state=42),     #0.59
    #"Gradient Boosting": GradientBoostingClassifier(random_state=42),         #0.78
    "XGBoost": XGBClassifier( eval_metric='logloss', random_state=42),         #0.96
    #"Neural Network": MLPClassifier(random_state=42),                          #0.55
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  # 0.96
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}")
#    
#    # Plot confusion matrix
#    cm = confusion_matrix(y_test, y_test_pred)
#    plt.figure(figsize=(6, 5))
#    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
#    plt.xlabel('Predicted')
#    plt.ylabel('Actual')
#    plt.title(f'Confusion Matrix - {model_name}')
#    plt.show()
#    
#    # Plot feature importances if supported
#    if hasattr(model, "feature_importances_"):
#        plt.figure(figsize=(10, 8))
#        importances = model.feature_importances_
#        indices = np.argsort(importances)[::-1]
#        top_features = 10  # Choose how many top features to display
#
#        # Only plot the top features for clarity
#        plt.bar(range(top_features), importances[indices[:top_features]], align='center')
#        plt.xticks(range(top_features), np.array(feature_cols)[indices[:top_features]], rotation=90)
#        plt.title(f'Top {top_features} Feature Importances - {model_name}')
#        plt.show()