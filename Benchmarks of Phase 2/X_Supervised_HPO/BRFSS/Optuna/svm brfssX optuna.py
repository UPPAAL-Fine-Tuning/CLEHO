import pandas as pd
import numpy as np
import optuna
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


X_train = pd.read_csv('X_train_brfss.csv')
X_test = pd.read_csv('X_test_brfss.csv')
y_train = pd.read_csv('y_train_brfss.csv').values.ravel() 
y_test = pd.read_csv('y_test_brfss.csv').values.ravel()
RANDOM_SEED = 42 


def objective_svm_rbf_balanced(trial):
    
    c_space = np.linspace(1, 500, 100).tolist()
    
    
    C_selected = trial.suggest_categorical("C", c_space)

    model = SVC(
        C=C_selected,  
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )

    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


print(f"\n{'='*60}")
print(f" Starting Optuna for SVM RBF (Balanced Weights)...")
print(f"Features: {X_train.shape[1]} original features")
print(f"{'='*60}")

study_rbf_bal = optuna.create_study(direction="maximize")
start_time = time.time()


study_rbf_bal.optimize(objective_svm_rbf_balanced, n_trials=15, show_progress_bar=True)

elapsed_time = time.time() - start_time


best_C_bal = study_rbf_bal.best_params['C']

final_model_rbf_bal = SVC(
    C=best_C_bal,
    kernel='rbf',
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=RANDOM_SEED
)
final_model_rbf_bal.fit(X_train, y_train)


y_pred_bal = final_model_rbf_bal.predict(X_test)
y_proba_bal = final_model_rbf_bal.predict_proba(X_test)[:, 1]


tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_bal).ravel()
acc_bal = accuracy_score(y_test, y_pred_bal)
prec_bal = precision_score(y_test, y_pred_bal, zero_division=0)
rec_bal = recall_score(y_test, y_pred_bal, zero_division=0)
f1_bal = f1_score(y_test, y_pred_bal, zero_division=0)
auc_bal = roc_auc_score(y_test, y_proba_bal)
spec_bal = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
g_mean_bal = np.sqrt(rec_bal * spec_bal)


print("\n" + "█"*60)
print(" FULL PERFORMANCE REPORT: SVM RBF (BALANCED)")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best C Found               : {best_C_bal:.6f}")
print("-" * 60)
print(f" Overall Accuracy          : {acc_bal:.4%}")
print(f" Sensitivity (Recall)      : {rec_bal:.4f}")
print(f" Specificity              : {spec_bal:.4f}")
print(f" F1-Score                  : {f1_bal:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean_bal:.4f}")
print(f" AUC-ROC Score             : {auc_bal:.4f}")
print("-" * 60)
print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_bal, target_names=['Healthy (0)', 'Diabetes (1)']))