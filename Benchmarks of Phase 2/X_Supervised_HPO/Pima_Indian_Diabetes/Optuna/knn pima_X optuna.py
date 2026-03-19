import pandas as pd
import optuna
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
y_train = pd.read_csv('pimaY_train.csv').values.ravel() 
y_test = pd.read_csv('pimaY_test.csv').values.ravel()

RANDOM_SEED = 42


def objective_knn(trial):
    
    n_neighbors = trial.suggest_int("n_neighbors", 1, 30)

    
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance', 
        n_jobs=-1
    )

    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


study_knn = optuna.create_study(direction="maximize")
start_time = time.time()

study_knn.optimize(objective_knn, n_trials=50, show_progress_bar=True)

total_optimization_time = time.time() - start_time


best_k = study_knn.best_params['n_neighbors']
final_model_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    n_jobs=-1
)
final_model_knn.fit(X_train, y_train)


y_pred = final_model_knn.predict(X_test)
y_proba = final_model_knn.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(rec * spec)


print("\n" + "█"*60)
print(" PERFORMANCE REPORT: KNN (PIMA DATASET)")
print("█"*60)
print(f"  Total Optimization Time   : {total_optimization_time:.2f}s")
print(f" Best n_neighbors (K)      : {best_k}")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f"  Specificity              : {spec:.4f}")
print(f"  F1-Score                 : {f1:.4f}")
print(f"  Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f"  AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f" Precision                 : {prec:.4f}")
print(f" True Positives (TP)      : {tp}")
print(f" True Negatives (TN)      : {tn}")
print(f"  False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n  FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))