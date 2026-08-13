import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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


c_dense_range = np.linspace(1, 500, 100).tolist() 
param_grid = {'C': c_dense_range}
model_name = 'SVM_RBF_Grid_Augmented'

print(f"\n{'='*60}")
print(f" Starting Grid Search for {model_name}...")
print(f"Feature Count: {X_train_aug.shape[1]} (63 + Cluster_Label)")
print(f" Testing {len(c_dense_range)} different values of C (Range: 1 to 500)...")
print(f" Total Model Fits: {len(c_dense_range) * 5} (100 params * 5 folds)")
print(f"{'='*60}")

# --- 3. Execute Grid Search ---
grid_search_svm = GridSearchCV(
    estimator=SVC(
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_SEED
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,  
    verbose=1   
)

start_time = time.time()
grid_search_svm.fit(X_train_aug, y_train_aug)
elapsed_time = time.time() - start_time


best_svm = grid_search_svm.best_estimator_
y_pred = best_svm.predict(X_test_aug)
y_proba = best_svm.predict_proba(X_test_aug)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test_aug, y_pred).ravel()
test_acc = accuracy_score(y_test_aug, y_pred)
test_rec = recall_score(y_test_aug, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_prec = precision_score(y_test_aug, y_pred, zero_division=0)
test_f1 = f1_score(y_test_aug, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test_aug, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


print("\n" + "█"*60)
print(f" FULL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best C found               : {grid_search_svm.best_params_['C']:.6f}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity              : {test_spec:.4f}")
print(f" F1-Score                  : {test_f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f"  AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f" Precision (Class 1)       : {test_prec:.4f}")
print(f"  True Positives (TP)      : {tp}")
print(f"  True Negatives (TN)      : {tn}")
print(f" False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test_aug, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))