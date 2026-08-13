from sklearn.model_selection import GridSearchCV

RANDOM_SEED = 42


n_estimators_range = np.unique(np.linspace(50, 500, 100).astype(int))
param_grid = {'n_estimators': n_estimators_range}

model_name = 'RF_GridSearch_Augmented'


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1 
)

print(f"\n{'='*60}")
print(f" Starting Dense Grid Search for {model_name}...")
print(f"Features: {X_train.shape[1]} (8 original + 1 Cluster_Label)")
print(f"{'='*60}")

start_time = time.time()
grid_search.fit(X_train, y_train)
elapsed_time = time.time() - start_time


best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_prec = precision_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


print("\n" + "█"*60)
print(f" FINAL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best n_estimators found   : {grid_search.best_params_['n_estimators']}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f"  Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity               : {test_spec:.4f}")
print(f"  F1-Score                 : {test_f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f" Precision (Class 1)       : {test_prec:.4f}")
print(f"  True Positives (TP)       : {tp}")
print(f" True Negatives (TN)       : {tn}")
print(f" False Positives (FP)      : {fp}")
print(f"  False Negatives (FN)      : {fn}")
print("-" * 60)

print("\n  FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))