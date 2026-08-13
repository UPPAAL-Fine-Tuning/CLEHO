from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


C_GRID = np.linspace(1, 500, 100)

 
def svm_objective_bayes(C):
    """
    Optimizing the regularization parameter C for the augmented dataset.
    """
    model = SVC(
        C=C,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',  
        probability=True,
        random_state=RANDOM_SEED
    )

    
    score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score


model_name = 'SVM_RBF_Bayesian_Augmented'
pbounds = {'C': (1, 500)}

optimizer = BayesianOptimization(
    f=svm_objective_bayes,
    pbounds=pbounds,
    random_state=RANDOM_SEED,
    verbose=2
)

print(f"\n{'='*60}")
print(f"  Starting Bayesian Optimization for {model_name}...")
print(f"Features: {X_train.shape[1]} (8 original + 1 Cluster_Label)")
print(f"{'='*60}")

start_time = time.time()

optimizer.maximize(init_points=5, n_iter=50)
elapsed_time = time.time() - start_time


raw_best_c = optimizer.max['params']['C']
best_c = C_GRID[np.abs(C_GRID - raw_best_c).argmin()]

print(f"\n🔍 Optimized Best C (Mapped to Grid): {best_c:.4f}")


final_model_svm = SVC(
    C=best_c,
    kernel='rbf',
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=RANDOM_SEED
)
final_model_svm.fit(X_train_scaled, y_train)


y_pred = final_model_svm.predict(X_test_scaled)
y_proba = final_model_svm.predict_proba(X_test_scaled)[:, 1]


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
test_acc = accuracy_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
test_auc = roc_auc_score(y_test, y_proba)
g_mean = np.sqrt(test_rec * test_spec)


print("\n" + "█"*60)
print(f" FINAL PERFORMANCE REPORT: {model_name}")
print("█"*60)
print(f" Total Optimization Time   : {elapsed_time:.2f}s")
print(f" Best C Parameter (Mapped) : {best_c:.4f}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f"  Sensitivity (Recall)      : {test_rec:.4f}")
print(f"  Specificity               : {test_spec:.4f}")
print(f"  F1-Score                 : {f1_score(y_test, y_pred):.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f"   True Positives (TP)      : {tp}")
print(f"   True Negatives (TN)      : {tn}")
print(f"  False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n  FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))