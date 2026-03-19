import pandas as pd
import numpy as np
import optuna
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


df_train = pd.read_csv('brfss_augmented_train.csv')
df_test = pd.read_csv('brfss_augmented_test.csv')

X_train_aug = df_train.drop(columns=['target'])
y_train_aug = df_train['target']

X_test_aug = df_test.drop(columns=['target'])
y_test_aug = df_test['target']

RANDOM_SEED = 42


def objective_svm_augmented(trial):
    
    c_options = np.linspace(1, 500, 100).tolist()
    
    
    c_value = trial.suggest_categorical("C", c_options)

    model = SVC(
        C=c_value,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )

    
    score = cross_val_score(model, X_train_aug, y_train_aug, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


print(f" Starting Optuna for SVM RBF on Augmented Data...")
print(f" Feature Count: {X_train_aug.shape[1]}")

study_svm = optuna.create_study(direction="maximize")
start_time = time.time()


study_svm.optimize(objective_svm_augmented, n_trials=15, show_progress_bar=True)

total_optimization_time = time.time() - start_time


best_c = study_svm.best_params['C']
final_model = SVC(
    C=best_c,
    kernel='rbf',
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=RANDOM_SEED
)
final_model.fit(X_train_aug, y_train_aug)


y_pred = final_model.predict(X_test_aug)
y_proba = final_model.predict_proba(X_test_aug)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test_aug, y_pred).ravel()
acc = accuracy_score(y_test_aug, y_pred)
rec = recall_score(y_test_aug, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
prec = precision_score(y_test_aug, y_pred)
f1 = f1_score(y_test_aug, y_pred)
auc = roc_auc_score(y_test_aug, y_proba)
g_mean = np.sqrt(rec * spec)


print("\n" + "█"*60)
print(" PERFORMANCE REPORT: OPTUNA SVM ON AUGMENTED DATA")
print("█"*60)
print(f" Total Optimization Time   : {total_optimization_time:.2f}s")
print(f" Best C found               : {best_c:.6f}")
print(f" Feature Count             : {X_train_aug.shape[1]}")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f"  F1-Score                  : {f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f" Precision                 : {prec:.4f}")
print(f"  True Positives (TP)      : {tp}")
print(f" True Negatives (TN)      : {tn}")
print(f"  False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n  FULL CLASSIFICATION REPORT:")
print(classification_report(y_test_aug, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))