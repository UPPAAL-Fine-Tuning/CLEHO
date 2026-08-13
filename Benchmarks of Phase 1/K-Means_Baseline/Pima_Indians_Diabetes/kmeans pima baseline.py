import pandas as pd
import numpy as np
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
X_kmeans = np.vstack((X_train, X_test))
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