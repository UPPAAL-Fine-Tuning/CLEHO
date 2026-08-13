
def knn_objective_augmented(n_neighbors):
    """
    Optimizing n_neighbors for the 64-feature augmented dataset.
    BayesOpt provides floats, so we cast to int for KNN.
    """
    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors),
        weights='distance',
        n_jobs=-1
    )

    
    accuracy = cross_val_score(model, X_train_aug, y_train_aug, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return accuracy


model_name = 'KNN_Bayesian_Augmented'

pbounds = {'n_neighbors': (1, 30)}

print(f"\n{'='*60}")
print(f"🚀 Starting Bayesian Optimization for {model_name}...")
print(f"🧬 Feature Count: {X_train_aug.shape[1]} (63 + Cluster_Label)")
print(f"{'='*60}")

optimizer = BayesianOptimization(
    f=knn_objective_augmented,
    pbounds=pbounds,
    random_state=RANDOM_SEED,
    verbose=2  
)

start_time = time.time()

optimizer.maximize(init_points=5, n_iter=10)
elapsed_time = time.time() - start_time


best_k = int(optimizer.max['params']['n_neighbors'])
print(f"\n  Best n_neighbors Found: {best_k}")

final_model_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', n_jobs=-1)
final_model_knn.fit(X_train_aug, y_train_aug)


y_pred = final_model_knn.predict(X_test_aug)
y_proba = final_model_knn.predict_proba(X_test_aug)[:, 1]


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
print(f" Best n_neighbors (K)      : {best_k}")
print("-" * 60)
print(f" Overall Accuracy          : {test_acc:.4%}")
print(f" Sensitivity (Recall)      : {test_rec:.4f}")
print(f" Specificity              : {test_spec:.4f}")
print(f" F1-Score                  : {test_f1:.4f}")
print(f" Geometric Mean (G-Mean)   : {g_mean:.4f}")
print(f" AUC-ROC Score             : {test_auc:.4f}")
print("-" * 60)
print(f" Precision (Class 1)       : {test_prec:.4f}")
print(f" True Positives (TP)      : {tp}")
print(f" True Negatives (TN)      : {tn}")
print(f" False Positives (FP)     : {fp}")
print(f"  False Negatives (FN)     : {fn}")
print("-" * 60)

print("\n FULL CLASSIFICATION REPORT:")
print(classification_report(y_test_aug, y_pred, target_names=['Healthy (0)', 'Diabetes (1)']))