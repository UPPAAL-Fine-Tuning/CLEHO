import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from bayes_opt import BayesianOptimization


RANDOM_SEED = 42


def knn_objective_bayes(n_neighbors):
    
    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors),
        weights='distance', 
        n_jobs=-1
    )
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


model_name = 'KNN_Bayesian_Diabetes_Augmented'
pbounds = {'n_neighbors': (1, 30)}


print(f"\n  Starting Bayesian Optimization for KNN ({X_train.shape[1]} features)...")
optimizer = BayesianOptimization(
    f=knn_objective_bayes,
    pbounds=pbounds,
    random_state=RANDOM_SEED,
    verbose=2
)

start_time = time.time()

optimizer.maximize(init_points=5, n_iter=50)
elapsed_time = time.time() - start_time


best_k = int(optimizer.max['params']['n_neighbors'])
final_model_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    n_jobs=-1
)
final_model_knn.fit(X_train, y_train)


y_pred = final_model_knn.predict(X_test)
y_proba = final_model_knn.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


print("\n" + "█"*60)
print(f" FINAL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f"  Best n_neighbors (K)      : {best_k}")
print(f"  Total Features Used       : {X_train.shape[1]}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity               : {test_spec:.4f}")
print(f" F1-Score                 : {f1_score(y_test, y_pred):.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f"  AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f"  True Positives (TP)      : {tp}")
print(f"  True Negatives (TN)      : {tn}")
print(f" False Positives (FP)     : {fp}")
print(f" False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))