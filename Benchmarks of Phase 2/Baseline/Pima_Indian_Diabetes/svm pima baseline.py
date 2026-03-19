import pandas as pd
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

X_train = pd.read_csv('pimaX_train.csv')
X_test = pd.read_csv('pimaX_test.csv')
y_train = pd.read_csv('pimaY_train.csv').values.ravel() 
y_test = pd.read_csv('pimaY_test.csv').values.ravel()

print(f" Data loaded.")
print(f" Number of features: {X_train.shape[1]}")

final_model_svm = SVC(
    C=1.0, 
    kernel='rbf',
    gamma='scale', 
    class_weight='balanced', 
    probability=True, 
    random_state=42
)

start_time = time.time()
final_model_svm.fit(X_train, y_train)
train_time = time.time() - start_time

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
print("PERFORMANCE REPORT: SVM RBF (PIMA DEFAULT)")
print("█"*60)
print(f"Training Time            : {train_time:.4f}s")
print(f"C Parameter (Default)     : 1.0")
print(f"Features Used             : {X_train.shape[1]}")
print("-" * 60)
print(f"Global Accuracy (Acc)     : {acc:.4%}")
print(f"Sensitivity (Recall)      : {rec:.4f}")
print(f"Specificity              : {spec:.4f}")
print(f"F1-Score                 : {f1:.4f}")
print(f"G-Mean                    : {g_mean:.4f}")
print(f"AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f"Precision (PPV)           : {prec:.4f}")
print(f"True Positives (TP)       : {tp}")
print(f"True Negatives (TN)       : {tn}")
print(f"False Positives (FP)      : {fp}")
print(f"False Negatives (FN)      : {fn}")
print("-" * 60)

print("\nFULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))