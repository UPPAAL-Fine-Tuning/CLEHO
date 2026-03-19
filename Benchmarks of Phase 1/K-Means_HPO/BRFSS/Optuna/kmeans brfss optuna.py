import pandas as pd
import numpy as np
import time
import optuna
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
X_kmeans = np.vstack((X_train, X_test))
RANDOM_SEED = 42

def optuna_obj(trial):
    k = trial.suggest_int("n_clusters", 2, 50)
    model = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    return silhouette_score(X_kmeans, labels)

print("\n🚀 Starting Optuna Study (10 Trials)...")
start_optuna = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(optuna_obj, n_trials=10)
elapsed_optuna = time.time() - start_optuna

best_k_opt = study.best_params['n_clusters']
opt_model = KMeans(n_clusters=best_k_opt, n_init=5, random_state=RANDOM_SEED).fit(X_kmeans)
labels = opt_model.labels_

print(f"\nMETHOD: Optuna")
print(f"K: {best_k_opt}")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, labels):.4f}")
print(f"INERTIA: {opt_model.inertia_:.2f}")
print(f"TIME (s): {elapsed_optuna:.2f}")