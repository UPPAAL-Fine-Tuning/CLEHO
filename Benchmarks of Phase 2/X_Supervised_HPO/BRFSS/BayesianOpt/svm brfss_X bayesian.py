import sys
import colorama
import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
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

c_space = np.linspace(1, 500, 100).tolist()

def svm_rbf_objective(C):
    idx = (np.abs(np.array(c_space) - C)).argmin()
    snapped_c = float(c_space[idx])
    
    model = SVC(
        C=snapped_c,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score

print("Starting Bayesian Optimization on Linear Scale C...")
print("Range: 1.0 to 500.0 (Snapped to 100 points)")

pbounds = {'C': (1, 500)}

optimizer = BayesianOptimization(
    f=svm_rbf_objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2 
)

start_time = time.time()
optimizer.maximize(init_points=5, n_iter=10)
elapsed_time = time.time() - start_time

raw_best_c = optimizer.max['params']['C']
idx = (np.abs(np.array(c_space) - raw_best_c)).argmin()
best_snapped_C = float(c_space[idx])

print(f"\n  Optimization Finished!")
print(f"Best Snapped C: {best_snapped_C:.6f}")

final_model = SVC(
    C=best_snapped_C,
    kernel='rbf',
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
g_mean = np.sqrt(rec * spec)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "█"*60)
print(" PERFORMANCE REPORT: BAYESIAN SVM (63 FEATURES)")
print("█"*60)
print(f" Total Time                : {elapsed_time:.2f}s")
print(f" Optimized C               : {best_snapped_C:.6f}")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f" True Positives (TP)       : {tp}")
print(f" True Negatives (TN)       : {tn}")
print(f" False Positives (FP)      : {fp}")
print(f" False Negatives (FN)      : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))