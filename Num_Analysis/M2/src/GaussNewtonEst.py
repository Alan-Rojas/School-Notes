"""
Pseudocode:
The idea here is to have a given model for estimating linearity propposed by a paper on a case study on this example. 
$$
y_i = delta + \fracc{\alpha-\delta}{1+e^{-\gamma(x_i-\beta)}} + \epsilon_i
$$
Meaning we have to estimate four parameters. 
The core idea here is to leverage the Gauss-Newton method given this model.

Step 1: 
We will initiate both the beta vector and the Z matrix.

Step 2:
We will then initiate the Differential matrix x, by comparing y to our model. So y - y_hat in essence.

Step 3:
Estimate the delta_beta, and update the beta by adding the delta_beta.

Step 4:
Given hyper-parameters for max iterations, convergence and tolerance, we start iterating. 

Again, getting the Z gradient matrix. -> Get the differential matrix D. --> Estimate the beta increment. 
--> Evaluate the convergence for new_beta and past-iteration beta. --> If converged, return beta_new.
--> Update beta with beta_new.  

Now, to do this, we need several attributes for our class:
To make this general, we will make the model customizable by making it a funcion entry.
We will also need x and y, with their respective comparisons so they are of the same shape.
We need init, fit and predict methods. 
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


class GaussNewtonApprox:
    def __init__(self, init_coef):
        """
        Init the estimator. X and Y attributes are set to none as they are to be given with the fit method.
        The model, which for simplicity is:
        $$
        y_i = delta + \fracc{\alpha-\delta}{1+e^{-\gamma(x_i-\beta)}} + \epsilon_i
        $$
        returns None.  
        """
        self.x = None
        self.y = None
        self.params = init_coef
        self.converged = False

    def func(self, x: np.array, params: np.array):
        """
        De acuerdo con Filstrup et al. (2014) 
        la clorofila A se puede modelar en función de otros nutrientes, particularmente:
        el fósforo total, con un modelo logístico de 4 parámetros.
        """
        return params[2] + (params[0]-params[2])/(1 + np.exp(-params[3]*(x-params[1])))

    def gradient(self, x, params):
        """
        Estimates the gradient of the function.
        """

        alpha, beta, delta, gamma = params

        f0 = 1/(1 + np.exp(-gamma*(x - beta))) # Alpha part. der
        f1 = (gamma * (alpha - delta)*np.exp(-gamma*(x - beta)))/(np.exp(-gamma*(x - beta)) + 1)**2
        f2 = 1 - 1/(1 + np.exp(-gamma*(x - beta)))
        f3 = ((alpha - delta)*(beta - x)*np.exp(-gamma*(x-beta)))/(1 + np.exp(-gamma*(x - beta)))**2

        return np.column_stack([f0, f1, f2, f3]) # Z matrix.
        
    def fit(self, x, y, max_iter = 1000, tol = 1e-6):
        self.x = x
        self.y = y

        i = 0

        while not self.converged and i < max_iter:
                ## Obtenemos la matriz de derivadas
            Z = self.gradient(x, self.params)

            ## Obtenemos la matriz de diferencias
            D = y - self.func(x, self.params)

            ## Obtenemos los estimadores del incremento de beta
            delta_params = np.linalg.solve(Z.T @ Z, Z.T @ D)

            ## Obtenemos la actualización de los coeficientes
            new_params = self.params + delta_params

            ## Revisamos la convergencia
            if np.abs(self.params - new_params).max() < tol:
                self.converged = True
                

            ## Actualizamos los coeficientes
            self.params = new_params
            i += 1

    def predict(self, x):
        """
        Predicts the given x using the function and the fine-tunned params.
        """

        return self.func(x, self.params)


class GaussNewtonAI:
    def __init__(self, max_iter=1000, tol=1e-6):
        """
        Initializes the estimator.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.converged = False
        self.params = None 
        self.used_fallback = False # Flag to track if curve_fit was used
        
    def _sigmoid_stable(self, z):
        """Numerically stable sigmoid function."""
        z = np.asarray(z)
        out = np.zeros_like(z, dtype=float)
        
        mask_pos = z >= 0
        out[mask_pos] = 1 / (1 + np.exp(-z[mask_pos]))
        
        mask_neg = z < 0
        out[mask_neg] = np.exp(z[mask_neg]) / (1 + np.exp(z[mask_neg]))
        
        return out

    def func(self, x: np.ndarray, params: list) -> np.ndarray:
        """Calculates the 4-parameter logistic model."""
        alpha, beta, delta, gamma = params
        z = gamma * (x - beta)
        return delta + (alpha - delta) * self._sigmoid_stable(z)

    def gradient(self, x: np.ndarray, params: list) -> np.ndarray:
        """Estimates the Jacobian matrix."""
        alpha, beta, delta, gamma = params
        z = gamma * (x - beta)
        
        S = self._sigmoid_stable(z)
        dS_dz = S * (1 - S)
        
        d_alpha = S
        d_beta = (alpha - delta) * dS_dz * (-gamma)
        d_delta = 1 - S
        d_gamma = (alpha - delta) * dS_dz * (x - beta)

        return np.column_stack([d_alpha, d_beta, d_delta, d_gamma])
        
    def _fallback_fit(self, initial_guesses):
        """
        Private method that uses scipy.optimize.curve_fit when pure 
        Gauss-Newton fails due to numerical instability.
        """
        self.used_fallback = True
        
        # curve_fit expects f(x, *params), so we wrap our func
        def wrapper_func(x_val, alpha, beta, delta, gamma):
            return self.func(x_val, [alpha, beta, delta, gamma])
        
        # Apply the exact bounds requested by the task
        lower_bounds = [0, np.min(self.x), 0, 0]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        
        try:
            popt, _ = curve_fit(
                wrapper_func, 
                self.x, 
                self.y, 
                p0=initial_guesses, 
                bounds=(lower_bounds, upper_bounds)
            )
            self.params = popt
            self.converged = True
            print("INFO: Gauss-Newton failed. Recovered successfully using 'curve_fit' fallback.")
        except RuntimeError as e:
            print(f"CRITICAL: Both Gauss-Newton and fallback optimization failed: {e}")

    def fit(self, x: np.ndarray, y: np.ndarray, init: list = None):
        """
        Fits the non-linear model. Tries Gauss-Newton first, falls back 
        to curve_fit if a singular matrix is encountered.
        """
        self.x = np.asarray(x).flatten()
        self.y = np.asarray(y).flatten()
        self.converged = False
        self.used_fallback = False
        
        # 1. Initialize parameters
        if init is None:
            alpha_0 = np.max(self.y)
            beta_0 = np.median(self.x)
            delta_0 = np.min(self.y)
            gamma_0 = 1.0
            self.params = np.array([alpha_0, beta_0, delta_0, gamma_0], dtype=float)
        else:
            self.params = np.array(init, dtype=float)

        initial_guesses = self.params.copy()
        fallback_needed = False
        i = 0

        # 2. Pure Gauss-Newton Loop
        while not self.converged and i < self.max_iter:
            Z = self.gradient(self.x, self.params)
            D = self.y - self.func(self.x, self.params)

            try:
                delta_params = np.linalg.solve(Z.T @ Z, Z.T @ D)
            except np.linalg.LinAlgError:
                # Catch the singular matrix error silently and flag for fallback
                fallback_needed = True
                break

            new_params = self.params + delta_params

            if np.max(np.abs(new_params - self.params)) < self.tol:
                self.converged = True
                
            self.params = new_params
            i += 1

        # Check if we hit max iterations without converging
        if not self.converged and not fallback_needed:
            fallback_needed = True

        # 3. Trigger Fallback if necessary
        if fallback_needed:
            self.params = initial_guesses # Reset to start cleanly
            self._fallback_fit(initial_guesses)
            
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the y values."""
        if self.params is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.func(np.asarray(x).flatten(), self.params)      



        