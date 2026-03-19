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

print("\n🚀 Starting Grid Search (K=2 to K=50)...")
start_grid = time.time()
grid_best_score = -1
grid_best_metrics = {}

for k in range(2, 50):
    model = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    sil = silhouette_score(X_kmeans, labels)
    
    if sil > grid_best_score:
        grid_best_score = sil
        grid_best_metrics = {
            'k': k,
            'silhouette': sil,
            'calinski': calinski_harabasz_score(X_kmeans, labels),
            'davies_bouldin': davies_bouldin_score(X_kmeans, labels),
            'inertia': model.inertia_
        }

elapsed_grid = time.time() - start_grid

print(f"\nMETHOD: GridSearch")
print(f"K: {grid_best_metrics['k']}")
print(f"SILHOUETTE: {grid_best_metrics['silhouette']:.4f}")
print(f"CALINSKI-H: {grid_best_metrics['calinski']:.2f}")
print(f"DAVIES-B: {grid_best_metrics['davies_bouldin']:.4f}")
print(f"INERTIA: {grid_best_metrics['inertia']:.2f}")
print(f"TIME (s): {elapsed_grid:.2f}")