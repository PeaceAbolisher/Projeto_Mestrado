import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

data = pd.read_csv(r"C:\\Users\\Rafael Fonseca\\Desktop\\Mestrado\\Ano2\\ProjetoMestrado\\parte_2\\data\\all_data_samples\\all_data_samples.csv", encoding="ISO-8859-1")

#separar features dos targets
X = data[['ncbi_taxon_id', 'scientific_name', 'relative_abundance']]
y = data['healthy']

#One-Hot Encode categorical columns
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(X[['ncbi_taxon_id', 'scientific_name']]).toarray()   #features: 2^14

X_combined = np.hstack((encoded_features, X[['relative_abundance']].values))

kf = KFold(n_splits=4, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

all_results = {name: [] for name in models}
all_results['CNN'] = []
all_results['MLP'] = []

for train_index, test_index in kf.split(X_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for name, model in models.items():
        model.fit(X_train, y_train)
        all_results[name].append((accuracy_score(y_train, model.predict(X_train)), accuracy_score(y_test, model.predict(X_test))))

    cnn = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=32, verbose=0)
    all_results['CNN'].append((accuracy_score(y_train, (cnn.predict(X_train.reshape(-1, X_train.shape[1], 1)) > 0.5).astype('int32')),
                               accuracy_score(y_test, (cnn.predict(X_test.reshape(-1, X_test.shape[1], 1)) > 0.5).astype('int32'))))

    mlp = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    all_results['MLP'].append((accuracy_score(y_train, (mlp.predict(X_train) > 0.5).astype('int32')),
                               accuracy_score(y_test, (mlp.predict(X_test) > 0.5).astype('int32'))))

# Print mean accuracy
for model_name, scores in all_results.items():
    mean_train = np.mean([train for train, _ in scores])
    mean_test = np.mean([test for _, test in scores])
    print(f"{model_name}: Mean Training Accuracy: {mean_train:.4f}, Mean Test Accuracy: {mean_test:.4f}")
