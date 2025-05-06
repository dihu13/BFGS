# BFGS vs. IRLS for Logistic Regression in R

This project benchmarks two optimization approaches—**BFGS** and **IRLS**—for fitting logistic regression models using Rcpp and RcppArmadillo. We implement both methods from scratch and evaluate their convergence behavior and performance on the CDC BRFSS 2020 heart disease dataset (~320,000 observations).

##  Project Structure

### 1. `BFGS_functions.cpp`
- Full Rcpp implementation of the BFGS optimization method for logistic regression.
- Includes:
  - `logli()`: Bernoulli log-likelihood.
  - `d1_logli()`: Gradient of the log-likelihood.
  - `Hk_f()`: BFGS Hessian update formula.
  - `optim_bfgs()`: BFGS algorithm with Wolfe condition-based line search.
  - `beta_se()`: Standard error calculation based on final Hessian.

### 2. `IRLS_functions.cpp`
- Rcpp implementation of the **Iteratively Reweighted Least Squares (IRLS)** algorithm.
- Includes:
  - `logli()`: Bernoulli log-likelihood (shared with BFGS).
  - `beta_updator()`: IRLS update step for β and standard errors.
  - `optim_irls()`: Full IRLS loop with log-likelihood-based convergence check.

### 3. `Benchmarking.R`
- Loads and preprocesses the CDC heart disease dataset (`heart_2020_cleaned.csv`).
- Fits logistic regression models using:
  - `glm()` (baseline),
  - `optim_irls()` (IRLS),
  - `optim_bfgs()` (BFGS).
- Compares coefficient estimates and convergence behavior across methods.
- Reports time taken and sensitivity to initial values.

##  Summary of Findings

- **IRLS** is typically faster than BFGS and converges in fewer iterations.
- **BFGS** can handle poor starting values more robustly than IRLS.
- Both methods yield similar coefficient estimates to the baseline `glm()` fit.

| Method | Speed | Robustness to Start Values | Accuracy |
|--------|-------|----------------------------|----------|
| IRLS   |  Fast |  May diverge with poor start |  High |
| BFGS   |  Slower |  More robust to start | High |
| GLM (`glm()`) |  Fast |  Stable |  High |

##  How to Run

1. Clone the repo.
2. Install the required R packages:
```r
install.packages(c("Rcpp", "RcppArmadillo"))
