import pandas as pd
import numpy as np
import time
import warnings
from bayes_opt import BayesianOptimization
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

def bayes_obj(n_clusters):
    k = int(round(n_clusters))
    model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    return silhouette_score(X_kmeans, labels)

print("\n Starting Bayesian Optimization (3 Init + 7 Iter = 10 Total)...")
start_time = time.time()
bo = BayesianOptimization(f=bayes_obj, pbounds={'n_clusters': (2, 50)}, random_state=RANDOM_SEED, verbose=2)
bo.maximize(init_points=3, n_iter=7)
elapsed = time.time() - start_time

best_k = int(round(bo.max['params']['n_clusters']))
model = KMeans(n_clusters=best_k, n_init=5, random_state=RANDOM_SEED).fit(X_kmeans)
labels = model.labels_

print(f"\nMETHOD: Bayesian")
print(f"K: {best_k}")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, labels):.4f}")
print(f"INERTIA: {model.inertia_:.2f}")
print(f"TIME (s): {elapsed:.2f}")