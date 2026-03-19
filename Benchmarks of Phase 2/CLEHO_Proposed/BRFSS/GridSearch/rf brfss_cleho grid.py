n_estimators_range = np.unique(np.linspace(50, 500, 100).astype(int))
param_grid = {'n_estimators': n_estimators_range}

model_name = 'RF_GridSearch_Augmented'

print(f"\n{'='*60}")
print(f"  Starting DENSE Grid Search for {model_name}...")
print(f"  Feature Count: {X_train_aug.shape[1]} (63 + Cluster_Label)")
print(f"  Testing {len(n_estimators_range)} unique tree configurations...")
print(f"  Total Fits: {len(n_estimators_range) * 5} (Params * 5 folds)")
print(f"{'='*60}")


grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search_rf.fit(X_train_aug, y_train_aug)
elapsed_time = time.time() - start_time


best_rf = grid_search_rf.best_estimator_
y_pred = best_rf.predict(X_test_aug)
y_proba = best_rf.predict_proba(X_test_aug)[:, 1]


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
print(f"  Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best n_estimators found   : {grid_search_rf.best_params_['n_estimators']}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f"  Specificity              : {test_spec:.4f}")
print(f" F1-Score                  : {test_f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f"  AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f"  Precision (Class 1)       : {test_prec:.4f}")
print(f"  True Positives (TP)      : {tp}")
print(f"   True Negatives (TN)      : {tn}")
print(f"  False Positives (FP)     : {fp}")
print(f" False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n  FULL CLASSIFICATION REPORT:")
print(classification_report(y_test_aug, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))