import pandas as pd
import optuna
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
y_train = pd.read_csv('pimaY_train.csv').values.ravel() 
y_test = pd.read_csv('pimaY_test.csv').values.ravel()

RANDOM_SEED = 42


N_ESTIMATORS_RANGE = np.unique(np.linspace(50, 500, 100).astype(int)).tolist()


def objective_rf_simple(trial):
    
    n_estimators = trial.suggest_categorical('n_estimators', N_ESTIMATORS_RANGE)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,          
        class_weight='balanced', 
        random_state=RANDOM_SEED,
        n_jobs=-1                
    )

    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


study_rf = optuna.create_study(direction="maximize")
start_time = time.time()


study_rf.optimize(objective_rf_simple, n_trials=50, show_progress_bar=True)

total_optimization_time = time.time() - start_time


best_n = study_rf.best_params['n_estimators']
final_model = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=None,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
final_model.fit(X_train, y_train)


y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(rec * spec)


print("\n" + "█"*60)
print(" PERFORMANCE REPORT: RANDOM FOREST (PIMA DATASET)")
print("█"*60)
print(f" Total Optimization Time   : {total_optimization_time:.2f}s")
print(f" Best n_estimators         : {best_n}")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f" F1-Score                 : {f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
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