#data augmentation of medium dataset
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import models, layers


sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Function to augment the dataset by adding noise to the feature columns
def augment_data_with_noise(data, target_size, noise_factor=0.01):
    augmented_data = data.copy()
    current_size = data.shape[0]
    
    while augmented_data.shape[0] < target_size:
        # Randomly sample rows from the original dataset
        sampled_data = data.sample(n=current_size, replace=True)
        
        # Apply random noise only to the feature columns (excluding the label column)
        features = sampled_data.iloc[:, :-1]  # Exclude the label column
        noise = np.random.normal(0, noise_factor, features.shape)
        noisy_features = features + noise  # Add noise to features
        
        # Combine noisy features with the original labels
        sampled_data.iloc[:, :-1] = noisy_features
        augmented_data = pd.concat([augmented_data, sampled_data], ignore_index=True)
    
    # Return the augmented dataset truncated to the target size
    return augmented_data[:target_size]

# Step 2: Load the original dataset
df = pd.read_csv('../medium_benchmark.csv')

# Augment the dataset to reach 2000 samples
df_augmented = augment_data_with_noise(df, target_size=2000)

# Optional: Save the augmented dataset (you can remove this if you don't need to save the file)
df_augmented.to_csv('augmented_dataset.csv', index=False)

# Step 3: Prepare the dataset (features and labels)
X = df_augmented.drop('Label', axis=1).values  # Features (OTU data)
y = df_augmented['Label'].values  # Labels (0 for healthy, 1 for T2D)

# Step 4: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define the CNN Model
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

# Step 6: Define the MLP Model
def create_mlp_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 7: Manual Cross-Validation with KFold
kf = KFold(n_splits=4, shuffle=True, random_state=42)

cnn_accuracies = []
mlp_accuracies = []

for train_index, val_index in kf.split(X_train_scaled):
    # Get training and validation data for this fold
    X_cv_train, X_cv_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]

    # Reshape data for CNN model
    X_cv_train_cnn = X_cv_train.reshape(X_cv_train.shape[0], X_cv_train.shape[1], 1)
    X_cv_val_cnn = X_cv_val.reshape(X_cv_val.shape[0], X_cv_val.shape[1], 1)

    # CNN Model
    cnn_model = create_cnn_model()
    cnn_model.fit(X_cv_train_cnn, y_cv_train, epochs=10, batch_size=32, verbose=0)
    cnn_pred = (cnn_model.predict(X_cv_val_cnn) > 0.5).astype("int32")
    cnn_acc = accuracy_score(y_cv_val, cnn_pred)
    cnn_accuracies.append(cnn_acc)

    # MLP Model
    mlp_model = create_mlp_model()
    mlp_model.fit(X_cv_train, y_cv_train, epochs=10, batch_size=32, verbose=0)
    mlp_pred = (mlp_model.predict(X_cv_val) > 0.5).astype("int32")
    mlp_acc = accuracy_score(y_cv_val, mlp_pred)
    mlp_accuracies.append(mlp_acc)

# Step 8: Print Cross-Validation Results
print(f'CNN Cross-Validation Accuracies: {cnn_accuracies}')
print(f'Average CNN Accuracy: {np.mean(cnn_accuracies):.4f}')

print(f'MLP Cross-Validation Accuracies: {mlp_accuracies}')
print(f'Average MLP Accuracy: {np.mean(mlp_accuracies):.4f}')
