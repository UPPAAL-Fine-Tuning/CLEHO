import pandas as pd
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

X_train_raw = pd.read_csv('train_features.csv')
X_test_raw = pd.read_csv('test_features.csv')
y_train_raw = pd.read_csv('train_labels.csv')
y_test_raw = pd.read_csv('test_labels.csv')

def preprocess_data(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            if col == 'Gender':
                df_encoded[col] = df_encoded[col].map({'Male': 1, 'Female': 0})
            else:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    return df_encoded

X_train = preprocess_data(X_train_raw)
X_test = preprocess_data(X_test_raw)

label_map = {'Positive': 1, 'Negative': 0}
y_train = y_train_raw.iloc[:, 0].map(label_map).values
y_test = y_test_raw.iloc[:, 0].map(label_map).values

print(f" Symptom data preprocessed.")
print(f" Number of symptoms analyzed: {X_train.shape[1]}")

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
print(" REPORT: SVM RBF (SYMPTOMS - DEFAULT)")
print("█"*60)
print(f" Training Time            : {train_time:.4f}s")
print(f" C Parameter (Default)     : 1.0")
print("-" * 60)
print(f" Global Accuracy (Acc)     : {acc:.4%}")
print(f" Sensitivity (Recall)      : {rec:.4f}")
print(f" Specificity              : {spec:.4f}")
print(f" F1-Score                 : {f1:.4f}")
print(f" AUC-ROC Score             : {auc:.4f}")
print("-" * 60)
print(f" True Positives (TP)       : {tp}")
print(f" True Negatives (TN)       : {tn}")
print(f" False Positives (FP)      : {fp}")
print(f" False Negatives (FN)      : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))