import pandas as pd
import numpy as np
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train_raw = pd.read_csv('train_features.csv')
X_test_raw = pd.read_csv('test_features.csv')

def preprocess_diabetes_data(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            if col == 'Gender':
                df_encoded[col] = df_encoded[col].map({'Male': 1, 'Female': 0})
            else:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    return df_encoded

X_kmeans = np.vstack((preprocess_diabetes_data(X_train_raw), preprocess_diabetes_data(X_test_raw)))
RANDOM_SEED = 42

print("\n Running Default K-Means (K=8)...")
start_time = time.time()
model = KMeans(n_clusters=8, n_init='auto', random_state=RANDOM_SEED) 
labels = model.fit_predict(X_kmeans)
elapsed = time.time() - start_time

print(f"METHOD: Default_K8")
print(f"K: 8")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, labels):.4f}")
print(f"INERTIA: {model.inertia_:.2f}")
print(f"TIME (s): {elapsed:.2f}")
