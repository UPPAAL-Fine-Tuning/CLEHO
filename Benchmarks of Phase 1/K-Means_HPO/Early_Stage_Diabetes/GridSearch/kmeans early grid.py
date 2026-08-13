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

print("\n Starting Grid Search (K=2 to K=50)...")
start_time = time.time()
best_score = -1
best_metrics = {}

for k in range(2, 51): 
    model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    sil = silhouette_score(X_kmeans, labels)
    
    if sil > best_score:
        best_score = sil
        best_metrics = {
            'k': k,
            'silhouette': sil,
            'calinski': calinski_harabasz_score(X_kmeans, labels),
            'davies_bouldin': davies_bouldin_score(X_kmeans, labels),
            'inertia': model.inertia_
        }

elapsed = time.time() - start_time

print(f"\nMETHOD: GridSearch")
print(f"K: {best_metrics['k']}")
print(f"SILHOUETTE: {best_metrics['silhouette']:.4f}")
print(f"CALINSKI-H: {best_metrics['calinski']:.2f}")
print(f"DAVIES-B: {best_metrics['davies_bouldin']:.4f}")
print(f"INERTIA: {best_metrics['inertia']:.2f}")
print(f"TIME (s): {elapsed:.2f}")