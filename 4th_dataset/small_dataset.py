import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import models, layers

sys.stdout.reconfigure(encoding='utf-8')

data = pd.read_csv('data/small_benchmark_with_extra_data.csv')

# Features and target
X = data.drop(columns=["Group", "Gender"])
y = data["Group"].map({"Healthy": 0, "T2D": 1}).values

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the CNN Model
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1], 1)))  # Corrected input shape
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Define the MLP Model
def create_mlp_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Manual Cross-Validation with KFold
kf = KFold(n_splits=4, shuffle=True, random_state=42)

cnn_accuracies = []
mlp_accuracies = []

for train_index, val_index in kf.split(X_train_scaled):
    X_cv_train, X_cv_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]

    # CNN Model
    X_cv_train_cnn = X_cv_train.reshape(X_cv_train.shape[0], X_cv_train.shape[1], 1)
    X_cv_val_cnn = X_cv_val.reshape(X_cv_val.shape[0], X_cv_val.shape[1], 1)
    cnn_model = create_cnn_model()
    cnn_model.fit(X_cv_train_cnn, y_cv_train, epochs=10, batch_size=32, verbose=0)
    cnn_pred = (cnn_model.predict(X_cv_val_cnn) > 0.5).astype("int32")
    cnn_accuracies.append(accuracy_score(y_cv_val, cnn_pred))

    # MLP Model
    mlp_model = create_mlp_model()
    mlp_model.fit(X_cv_train, y_cv_train, epochs=10, batch_size=32, verbose=0)
    mlp_pred = (mlp_model.predict(X_cv_val) > 0.5).astype("int32")
    mlp_accuracies.append(accuracy_score(y_cv_val, mlp_pred))

# Save results to a file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(f"CNN Cross-Validation Accuracies: {cnn_accuracies}\n")
    f.write(f"Mean CNN Accuracy: {np.mean(cnn_accuracies)}\n")
    f.write(f"MLP Cross-Validation Accuracies: {mlp_accuracies}\n")
    f.write(f"Mean MLP Accuracy: {np.mean(mlp_accuracies)}\n")