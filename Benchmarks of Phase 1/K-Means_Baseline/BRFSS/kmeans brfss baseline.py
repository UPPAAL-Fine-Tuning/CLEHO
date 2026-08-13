import pandas as pd
import numpy as np
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
X_kmeans = np.vstack((X_train, X_test))
RANDOM_SEED = 42

print("\n Running Default K-Means (Standard library defaults)...")
start_def = time.time()
default_model = KMeans(random_state=RANDOM_SEED) 
default_labels = default_model.fit_predict(X_kmeans)
elapsed_def = time.time() - start_def

print(f"METHOD: Default_K8")
print(f"K: 8")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, default_labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, default_labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, default_labels):.4f}")
print(f"INERTIA: {default_model.inertia_:.2f}")
print(f"TIME (s): {elapsed_def:.2f}")
