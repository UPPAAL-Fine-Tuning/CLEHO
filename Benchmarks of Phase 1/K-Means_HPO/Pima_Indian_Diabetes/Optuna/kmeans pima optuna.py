import pandas as pd
import numpy as np
import time
import optuna
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
X_kmeans = np.vstack((X_train, X_test))
RANDOM_SEED = 42

def optuna_obj(trial):
    k = trial.suggest_int("n_clusters", 2, 50)
    model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_SEED, algorithm='elkan')
    labels = model.fit_predict(X_kmeans)
    score = silhouette_score(X_kmeans, labels)
    print(f"   Trial {trial.number:2} | Testing K={k:2} | Silhouette: {score:.4f}")
    return score

print("\n🚀 Starting Optuna Study (10 Trials)...")
study = optuna.create_study(direction="maximize")
start_time = time.time()
study.optimize(optuna_obj, n_trials=10)
elapsed = time.time() - start_time

best_k = study.best_params['n_clusters']
model = KMeans(n_clusters=best_k, n_init=5, random_state=RANDOM_SEED).fit(X_kmeans)
labels = model.labels_

print(f"\nMETHOD: Optuna")
print(f"K: {best_k}")
print(f"SILHOUETTE: {silhouette_score(X_kmeans, labels):.4f}")
print(f"CALINSKI-H: {calinski_harabasz_score(X_kmeans, labels):.2f}")
print(f"DAVIES-B: {davies_bouldin_score(X_kmeans, labels):.4f}")
print(f"INERTIA: {model.inertia_:.2f}")
print(f"TIME (s): {elapsed:.2f}")