import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from bayes_opt import BayesianOptimization


df_train = pd.read_csv('brfss_augmented_train.csv')
df_test = pd.read_csv('brfss_augmented_test.csv')

X_train_aug = df_train.drop(columns=['target'])
y_train_aug = df_train['target']

X_test_aug = df_test.drop(columns=['target'])
y_test_aug = df_test['target']


c_dense_range = np.linspace(1, 500, 100).tolist()


def svm_rbf_objective_augmented(c_index):
    """
    Objective function that maps the optimizer's continuous choice 
    to the nearest index in your 100-value list.
    """
    
    idx = int(round(c_index))
    actual_C = c_dense_range[idx]

    model = SVC(
        C=actual_C,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    
    score = cross_val_score(model, X_train_aug, y_train_aug, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


print(f"🚀 Starting Bayesian Optimization on Augmented Data...")
print(f"🧬 Feature Count: {X_train_aug.shape[1]} (63 Features + Cluster)")
print(f"📏 Search Space: Picking from 100 discrete values between 1 and 500")


pbounds = {'c_index': (0, 99)}

optimizer = BayesianOptimization(
    f=svm_rbf_objective_augmented,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

start_time = time.time()
optimizer.maximize(init_points=5, n_iter=15) 
total_optimization_time = time.time() - start_time


best_idx = int(round(optimizer.max['params']['c_index']))
best_actual_C = c_dense_range[best_idx]

print(f"\n🏆 Best Hyperparameters Found:")
print(f"   - Selected Index: {best_idx}")
print(f"   - Optimized C: {best_actual_C:.4f}")

final_model = SVC(
    C=best_actual_C,
    kernel='rbf',
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
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
print(" PERFORMANCE REPORT: BAYESIAN SVM RBF (AUGMENTED)")
print("█"*60)
print(f"  Total Optimization Time   : {total_optimization_time:.2f}s")
print(f" Best C Parameter          : {best_actual_C:.6f}")
print(f"  Feature Count             : {X_train_aug.shape[1]} (63 original + 1 cluster)")
print("-" * 60)
print(f" Overall Accuracy          : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f" F1-Score                  : {f1:.4f}")
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
print(classification_report(y_test_aug, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))