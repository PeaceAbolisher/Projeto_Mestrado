import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import models, layers

sys.stdout.reconfigure(encoding='utf-8')


def augment_data_with_noise(data, target_size, noise_factor=0.01):
    augmented_data = data.copy()
    current_size = data.shape[0]
    
    while augmented_data.shape[0] < target_size:
        sampled_data = data.sample(n=current_size, replace=True)
        
        features = sampled_data.iloc[:, :-1]  
        noise = np.random.normal(0, noise_factor, features.shape)
        noisy_features = features + noise 
        
        sampled_data.iloc[:, :-1] = noisy_features
        augmented_data = pd.concat([augmented_data, sampled_data], ignore_index=True)
    
    return augmented_data[:target_size]


df = pd.read_csv('data/medium_benchmark.csv')

df_augmented = augment_data_with_noise(df, target_size=2000)

df_augmented.to_csv('augmented_dataset.csv', index=False)

X = df_augmented.drop('Label', axis=1).values 
y = df_augmented['Label'].values  #(0 for healthy, 1 for T2D)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_mlp_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


kf = KFold(n_splits=4, shuffle=True, random_state=42)

cnn_accuracies = []
mlp_accuracies = []

for train_index, val_index in kf.split(X_train_scaled):
    X_cv_train, X_cv_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]
    
    # Reshape for CNN: (samples, features, channels)
    X_cv_train_cnn = X_cv_train.reshape(X_cv_train.shape[0], X_cv_train.shape[1], 1)
    X_cv_val_cnn = X_cv_val.reshape(X_cv_val.shape[0], X_cv_val.shape[1], 1)
    
    # --- CNN Model ---
    cnn_model = create_cnn_model()
    cnn_model.fit(X_cv_train_cnn, y_cv_train, epochs=10, batch_size=32, verbose=0)
    cnn_pred = (cnn_model.predict(X_cv_val_cnn) > 0.5).astype("int32")
    cnn_acc = accuracy_score(y_cv_val, cnn_pred)
    cnn_accuracies.append(cnn_acc)
    
    # --- MLP Model ---
    mlp_model = create_mlp_model()
    mlp_model.fit(X_cv_train, y_cv_train, epochs=10, batch_size=32, verbose=0)
    mlp_pred = (mlp_model.predict(X_cv_val) > 0.5).astype("int32")
    mlp_acc = accuracy_score(y_cv_val, mlp_pred)
    mlp_accuracies.append(mlp_acc)

avg_cnn_accuracy = np.mean(cnn_accuracies)
avg_mlp_accuracy = np.mean(mlp_accuracies)

#Evaluation on the Test Set

#Reshape the full training and test sets for the CNN model
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# --- Final CNN Model ---
final_cnn_model = create_cnn_model()
final_cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=0)
cnn_test_pred = (final_cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
cnn_test_acc = accuracy_score(y_test, cnn_test_pred)

# --- Final MLP Model ---
final_mlp_model = create_mlp_model()
final_mlp_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
mlp_test_pred = (final_mlp_model.predict(X_test_scaled) > 0.5).astype("int32")
mlp_test_acc = accuracy_score(y_test, mlp_test_pred)


print("\n=== Cross-Validation Results ===")
print(f"  CNN Accuracies per fold: {cnn_accuracies}")
print(f"  Average CNN Accuracy: {avg_cnn_accuracy:.4f}")
print(f"  MLP Accuracies per fold: {mlp_accuracies}")
print(f"  Average MLP Accuracy: {avg_mlp_accuracy:.4f}\n")

print("\n=== Final Test Set Results ===")
print(f"  CNN Test Accuracy: {cnn_test_acc:.4f}")
print(f"  MLP Test Accuracy: {mlp_test_acc:.4f}")
