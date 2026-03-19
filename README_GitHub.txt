# Can Hyperparameter-Optimized Unsupervised Learning Improve Supervised Models for Diabetes Detection?

This repository contains the official implementation, formal models, and experimental benchmarks for the **CLEHO** framework. CLEHO bridges the gap between unsupervised structural discovery and supervised diagnostic prediction through a dual-stage HPO approach.

## Framework Overview
CLEHO operates in two synchronized phases:
1. **Phase 1 (Unsupervised):** Optimizes data geometry using HPO to identify natural clinical sub-phenotypes (C_{opt}).
2. **Phase 2 (Supervised):** Leverages the enriched feature space X' = [X, C_{opt}] to tune high-performance classifiers, reaching near-perfect diagnostic accuracy.

---

## 📂 Repository Structure

The project is organized to mirror the methodology described in the manuscript:

### 📁 Benchmarks of Phase 1 (Unsupervised HPO)
Focuses on formalizing the clustering process.
* K-Means_HPO/: Contains scripts for navigating the search space Omega_{unsup} (k in [2, 50]).
  * Sub-folders: Grid_Search, Bayesian_Opt, Optuna, and Proposed_Method.
  * Metrics: Silhouette Score, Davies-Bouldin Index, and Inertia analysis.

### 📁 Benchmarks of Phase 2 (Supervised HPO)
Focuses on diagnostic performance and comparative analysis.
* Baseline/: Standard classification on raw clinical data (X).
* X_Supervised_HPO/: Standard HPO without structural enrichment.
* CLEHO_Proposed/: The complete dual-stage pipeline using optimized cluster-based enrichment.
  * Includes performance comparisons (Accuracy).

---

### Prerequisites
* Python 3.8+
* Libraries: pandas, scikit-learn, optuna, matplotlib, seaborn

### Reproduction
To replicate the results for the Early Stage Diabetes, Pima Indians or the large-scale BRFSS datasets:
Phase 1 (Structural Discovery): Navigate to Benchmarks of Phase 1/K-Means_HPO/. 
Run the scripts to navigate the search space \Omega_{unsup} and generate the optimized cluster labels (C_{opt}).
Phase 2 (Diagnostic Prediction): Use the generated C_{opt} as an additional feature to enrich your dataset. 
Navigate to Benchmarks of Phase 2/CLEHO_Proposed/ and run the scripts to execute the dual-stage HPO and reach the reported diagnostic peaks.

---

## 📊 Key Results
Early Stage Diabetes (High Precision): Achieved a near-perfect 99.98% accuracy using the CLEHO framework, setting a new benchmark for automated risk prediction.
Pima Indians Diabetes (Robustness): Demonstrated significant performance stability and superior diagnostic alignment compared to state-of-the-art hybrid models.
BRFSS Dataset (Scalability): Validated the framework on a massive epidemiological scale (>250,000 instances), achieving a structural gain of +343.2% in Silhouette score compared to default heuristics.

---

## License & Citation
This code is provided for research purposes. If you use this framework, please cite:
> S. Ben Ahmed, et al. Can Hyperparameter-Optimized Unsupervised Learning Improve Supervised Models for Diabetes Detection?... (202x).

##Contact: sirine.benahmed@esgitech.tn