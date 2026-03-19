import pandas as pd
import optuna
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
y_train = pd.read_csv('pimaY_train.csv').values.ravel() 
y_test = pd.read_csv('pimaY_test.csv').values.ravel()

RANDOM_SEED = 42


C_RANGE = np.linspace(1, 500, 100).tolist()

def objective_svm(trial):
    
    c_value = trial.suggest_categorical('C', C_RANGE)

    model = SVC(
        C=c_value,
        kernel='rbf',
        gamma='scale',           
        class_weight='balanced', 
        probability=True,        
        random_state=RANDOM_SEED
    )

    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


study_svm = optuna.create_study(direction="maximize")
start_time = time.time()

study_svm.optimize(objective_svm, n_trials=50, show_progress_bar=True)

total_optimization_time = time.time() - start_time


best_c = study_svm.best_params['C']
final_model_svm = SVC(
    C=best_c,
    kernel='rbf',
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=RANDOM_SEED
)
final_model_svm.fit(X_train, y_train)


y_pred = final_model_svm.predict(X_test)
y_proba = final_model_svm.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(rec * spec)


print("\n" + "█"*60)
print(" PERFORMANCE REPORT: SVM RBF (PIMA DATASET)")
print("█"*60)
print(f" Total Optimization Time   : {total_optimization_time:.2f}s")
print(f" Best C Parameter          : {best_c:.4f}")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f" F1-Score                 : {f1:.4f}")
print(f"  Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f" Precision                 : {prec:.4f}")
print(f" True Positives (TP)      : {tp}")
print(f" True Negatives (TN)      : {tn}")
print(f" False Positives (FP)     : {fp}")
print(f" False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))