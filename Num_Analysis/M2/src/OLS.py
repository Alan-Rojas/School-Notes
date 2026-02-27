import numpy as np

class OLS:
    def __init__(self):
        self.beta_hat = None
        self.L = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Generalize to handle both 1D (single variable) and 2D (multiple variables)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensiones incompatibles.")
            
        # Agregar columna de 1s (Intercepto)
        X_aug = np.column_stack((np.ones(X.shape[0]), X))
        
        # Computationally efficient solver instead of inverse
        # Resolving: (X^T X) * beta = X^T y
        self.beta_hat = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        
        # Cálculo del Error (Suma de errores al cuadrado)
        self.L = np.sum((y - X_aug @ self.beta_hat)**2)
        
        return self.beta_hat, self.L

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_aug = np.column_stack((np.ones(X.shape[0]), X))
        return X_aug @ self.beta_hat
    

class FGLS:
    """
    The purpose of this class is to have a class that can perform the FGLS algorithm for a given dataset.
    Generally, we would want to take an input like X (the predictor variable) and y (the target variable). 

    The suggested workflow for the FGLS algorithm is the following:

    1. Estimate beta_hat using OLS for time-stamp 0.
    2. For time-stamps $t = 0$ trought a max_iteration, perform:

        a. $e^{(t)}_i = (y_i - x^{T}_i \hat{beta}^{(t)})²$, for $i = 1, ..., n$

        b. $\hat{\gamma}^{(t)} = (X^T X)^{-1} X^T e^{(t)}$

        c. $w_i^{(t)} = \fracc{1}{X_i^T \hat{\gamma}^{(t)}}$

        d. $W^T = \diag(w_1^{(t)}, ..., w_n^{(t)})$

        e. $\hat{\beta}^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} y$

        f. if $||\hat{\Beta}^{(t+1)} - \hat{\Beta}^{(t)}||_p < tol$, return Beta, else go back to step a.

    """

    def __init__(self):
        self.beta_t = None
        self.converged = False

    def fit(self, X, y, max_iteration: int = 1000, tol: float = 1e-5):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X_aug = np.column_stack((np.ones(X.shape[0]), X))
        
        # Step 1: Getting Beta_0 with OLS
        beta_t = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)

        for i in range(max_iteration):
            # Step a: Vectorized squared residuals
            e_t = (y - X_aug @ beta_t)**2

            # Step b: Gamma to estimate variance
            gamma_t = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ e_t)

            # Step c: Calculate weights, were we punish high variance
            var_estimated = X_aug @ gamma_t
            
            # Safety mechanism: Variance cannot be negative or zero
            var_estimated = np.clip(var_estimated, a_min=1e-8, a_max=None)
            w = 1.0 / var_estimated

            # Step d & e: WLS using broadcasting (O(N) memory instead of O(N^2))
            # Mathematically equivalent to X^T @ W @ X, but drastically faster
            X_W = X_aug * w[:, np.newaxis] 
            
            # B_new = (X^T W X)^-1 (X^T W y)
            B_new = np.linalg.solve(X_W.T @ X_aug, X_W.T @ y)

            # Step f: Check convergence using the L2 norm
            error_diff = np.linalg.norm(B_new - beta_t, ord=2)
            
            beta_t = B_new # Update for next loop
            
            if error_diff < tol:
                self.converged = True
                print(f"Algorithm converged on {i}th iteration.")
                break
                
        self.beta_t = beta_t
        return self.beta_t

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_aug = np.column_stack((np.ones(X.shape[0]), X))
        return X_aug @ self.beta_t

def case_1(n: int, b_0: float, b_1: float, b_2: float, b_3: float):
    """
    x_i ~ iid ~ U(10, 100), i = 1, ..., n
    y_i ~ iid ~ N(mu_i, sigma2_i), i = 1, ..., n
    mu_i = b_0 + b1*x_i
    log(sigma2_i) = b_2 + b_3*x_i
    """
    # Set seed for reproducible results
    np.random.seed(42)
    
    # 1. Generate X
    X = np.random.uniform(10, 100, n)
    
    # 2. Calculate the True Parameters for the Normal Distribution
    mu = b_0 + b_1 * X
    
    # log(sigma2_i) = b_2 + b_3 * x_i  =>  sigma_i = sqrt(exp(b_2 + b_3 * x_i))
    sigma = np.sqrt(np.exp(b_2 + b_3 * X))
    
    # 3. Generate y
    y = np.random.normal(loc=mu, scale=sigma)
    
    # 4. Fit the Models
    ols = OLS()
    beta_ols, _ = ols.fit(X, y)
    
    fgls = FGLS()
    beta_fgls = fgls.fit(X, y)
    
    # 5. Evaluate the Results
    print("--- Algorithm Comparison ---")
    print(f"True Betas : Intercept = {b_0:.4f}, Slope = {b_1:.4f}")
    print(f"OLS Betas  : Intercept = {beta_ols[0]:.4f}, Slope = {beta_ols[1]:.4f}")
    print(f"FGLS Betas : Intercept = {beta_fgls[0]:.4f}, Slope = {beta_fgls[1]:.4f}")
    
    # Calculate absolute errors to see which model got closer
    err_ols = np.abs(beta_ols - np.array([b_0, b_1]))
    err_fgls = np.abs(beta_fgls - np.array([b_0, b_1]))
    
    print("\n--- Absolute Errors (Lower is better) ---")
    print(f"OLS Error  : Intercept = {err_ols[0]:.4f}, Slope = {err_ols[1]:.4f}")
    print(f"FGLS Error : Intercept = {err_fgls[0]:.4f}, Slope = {err_fgls[1]:.4f}")

    return X, y, beta_ols, beta_fgls
    
if __name__ == "__main__":
    # Case 1: n = 500, b_0 = 5, b_1 = 2, b_2 = 0.4, b_3 = -0.2 
    X, y, beta_ols, beta_fgls = case_1(n=500, b_0=5.0, b_1=2.0, b_2=0.4, b_3=-0.2)

