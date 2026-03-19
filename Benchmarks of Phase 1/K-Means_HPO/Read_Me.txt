# Phase 1: K-Means Hyperparameter Optimization

This stage focuses on finding the natural geometric structure of the clinical data. 
By optimizing the clustering process, we identify the most coherent sub-phenotypes (C_{opt}) to enrich the supervised models in Phase 2.

### Optimization Techniques Evaluated:
* **GridSearch**: Exhaustive exploration of the cluster search space.
* **Bayesian**: Sequential model-based optimization for efficient convergence.
* **Optuna**: TPE-based sampling for high-dimensional efficiency.
* **Proposed_Method**: Our specialized logic for identifying stable structural optima.