
import sys
import colorama
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

if not hasattr(colorama, 'just_fix_windows_console'):
    colorama.just_fix_windows_console = lambda: None

from bayes_opt import BayesianOptimization

X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
y_train = pd.read_csv('y_train_brfss.csv').values.ravel() 
y_test = pd.read_csv('y_test_brfss.csv').values.ravel()

RANDOM_SEED = 42

def knn_objective_bayes(n_neighbors):
    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors),
        weights='distance',
        n_jobs=-1
    )
    
    return cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()


model_name = 'KNN_Bayesian'
pbounds = {'n_neighbors': (1, 30)}

optimizer = BayesianOptimization(f=knn_objective_bayes, pbounds=pbounds, random_state=RANDOM_SEED, verbose=2)

start_time = time.time()
optimizer.maximize(init_points=5, n_iter=10)
elapsed_time = time.time() - start_time


best_k = int(optimizer.max['params']['n_neighbors'])
final_model_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', n_jobs=-1)
final_model_knn.fit(X_train, y_train)


y_pred = final_model_knn.predict(X_test)
y_proba = final_model_knn.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_prec = precision_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)

results_summary = {}
results_summary[model_name] = {
    'optimizer': 'Bayesian',
    'best_params': {'n_neighbors': best_k},
    'test_accuracy': test_acc,
    'test_precision': test_prec,
    'test_recall': test_rec,
    'test_f1_score': test_f1,
    'test_specificity': test_spec,
    'test_geometric_mean': g_mean,
    'test_auc_roc': test_auc,
    'optimization_time': elapsed_time
}


print("\n" + "█"*60)
print(f" FULL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f"  Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best n_neighbors (K)      : {best_k}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity              : {test_spec:.4f}")
print(f" F1-Score                  : {test_f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f" Precision (Class 1)       : {test_prec:.4f}")
print(f" True Positives (TP)      : {tp}")
print(f" True Negatives (TN)      : {tn}")
print(f" False Positives (FP)     : {fp}")
print(f" False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))