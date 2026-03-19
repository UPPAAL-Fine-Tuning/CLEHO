from bayes_opt import BayesianOptimization

RANDOM_SEED = 42


n_estimators_range = np.unique(np.linspace(50, 500, 100).astype(int))
lower_bound = n_estimators_range.min()
upper_bound = n_estimators_range.max()


def rf_objective_bayes(n_estimators):
    
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None,          
        class_weight='balanced', 
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return accuracy


model_name = 'RF_Bayesian_Diabetes_Augmented'
pbounds = {'n_estimators': (lower_bound, upper_bound)}


print(f"\n  Starting Bayesian Optimization for {X_train.shape[1]} features...")
optimizer = BayesianOptimization(
    f=rf_objective_bayes,
    pbounds=pbounds,
    random_state=RANDOM_SEED,
    verbose=2
)

start_time = time.time()

optimizer.maximize(init_points=5, n_iter=50)
elapsed_time = time.time() - start_time


raw_best_n = optimizer.max['params']['n_estimators']
best_n = int(n_estimators_range[np.abs(n_estimators_range - raw_best_n).argmin()])

final_model_rf = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=None,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
final_model_rf.fit(X_train, y_train)


y_pred = final_model_rf.predict(X_test)
y_proba = final_model_rf.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


print("\n" + "█"*60)
print(f" FINAL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f"  Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best n_estimators         : {best_n}")
print(f" Total Features Used       : {X_train.shape[1]}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f"  Sensitivity (Recall)      : {test_rec:.4f}")
print(f"  Specificity               : {test_spec:.4f}")
print(f" F1-Score                 : {f1_score(y_test, y_pred):.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f"  True Positives (TP)      : {tp}")
print(f"  True Negatives (TN)      : {tn}")
print(f"  False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))