from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
y_train = pd.read_csv('y_train_brfss.csv').values.ravel() 
y_test = pd.read_csv('y_test_brfss.csv').values.ravel()
RANDOM_SEED = 42 

results_summary = {}

k_range = list(range(1, 31))
param_grid = {'n_neighbors': k_range}

model_name = 'KNN_GridSearch_30'

print(f"\n{'='*60}")
print(f" Starting Grid Search for {model_name}...")
print(f"Testing K values: 1, 2, ..., 30")
print(f"Total Model Fits: {len(k_range) * 5}")
print(f"{'='*60}")


grid_search_knn = GridSearchCV(
    estimator=KNeighborsClassifier(weights='distance'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search_knn.fit(X_train, y_train)
elapsed_time = time.time() - start_time


best_knn = grid_search_knn.best_estimator_
y_pred = best_knn.predict(X_test)
y_proba = best_knn.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_prec = precision_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


results_summary[model_name] = {
    'optimizer': 'GridSearchCV',
    'best_params': grid_search_knn.best_params_,
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
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best K found               : {grid_search_knn.best_params_['n_neighbors']}")
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