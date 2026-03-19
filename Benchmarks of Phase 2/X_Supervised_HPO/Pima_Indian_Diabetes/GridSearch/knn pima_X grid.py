import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
y_train = pd.read_csv('pimaY_train.csv').values.ravel() 
y_test = pd.read_csv('pimaY_test.csv').values.ravel()

RANDOM_SEED = 42

param_grid_knn = {'n_neighbors': np.arange(1, 31)}

model_name = 'KNN_GridSearch_Unscaled'

print(f"\n{'='*60}")
print(f" Starting Grid Search for {model_name}...")
print(f"Testing K values from 1 to 30...")
print(f"{'='*60}")


grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(weights='distance', n_jobs=-1),
    param_grid=param_grid_knn,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()

grid_search.fit(X_train, y_train)
elapsed_time = time.time() - start_time


best_knn = grid_search.best_estimator_
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


print("\n" + "█"*60)
print(f" FULL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best n_neighbors (K) found: {grid_search.best_params_['n_neighbors']}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity              : {test_spec:.4f}")
print(f" F1-Score                 : {test_f1:.4f}")
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