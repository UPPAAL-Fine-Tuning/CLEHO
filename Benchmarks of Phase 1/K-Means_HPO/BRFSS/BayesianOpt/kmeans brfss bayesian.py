import pandas as pd
import numpy as np
import time
import warnings
from bayes_opt import BayesianOptimization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
X_kmeans = np.vstack((X_train, X_test))
RANDOM_SEED = 42

def bayes_obj(n_clusters):
    k = int(round(n_clusters))
    model = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    return silhouette_score(X_kmeans, labels)

print("\n🚀 Starting Bayesian Optimization (10 Total Points)...")
start_bayes = time.time()
bo = BayesianOptimization(f=bayes_obj, pbounds={'n_clusters': (2, 50)}, random_state=RANDOM_SEED)
bo.maximize(init_points=3, n_iter=7)
elapsed_bayes = time.time() - start_bayes

best_k_bay = int(round(bo.max['params']['n_clusters']))
bay_model = KMeans(n_clusters=best_k_bay, n_init=5, random_state=RANDOM_SEED).fit(X_kmeans)
labels = bay_model.labels_

print(f"\nMETHOD: Bayesian")
print(f"K: {best_k_bay}")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, labels):.4f}")
print(f"INERTIA: {bay_model.inertia_:.2f}")
print(f"TIME (s): {elapsed_bayes:.2f}")