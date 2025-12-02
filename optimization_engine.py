"""
Optimization Engine
==================
Handle optimization with constraint support and path tracking.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, List, Optional, Tuple


class OptimizationEngine:
    """Handle optimization with constraint support and path tracking."""

    def __init__(self, objective_func: Callable, constraint_func: Optional[Callable] = None,
                 num_vars: int = 2, bounds: List[Tuple[float, float]] = None):
        self.objective_func = objective_func
        self.constraint_func = constraint_func
        self.num_vars = num_vars
        self.bounds = bounds or [(0.01, 10.0)] * num_vars
        self.optimization_path = []

    def _objective_wrapper(self, x: np.ndarray) -> float:
        """Wrapper for objective function that handles arrays."""
        try:
            result = self.objective_func(*x)
            return float(result)
        except:
            return 1e10

    def _objective_with_penalty(self, x: np.ndarray, penalty_weight: float = 100000) -> float:
        """Objective with constraint penalty."""
        obj_val = self._objective_wrapper(x)

        if self.constraint_func is not None:
            try:
                constraint_val = self.constraint_func(*x)
                penalty = penalty_weight * constraint_val ** 2
                obj_val += penalty
            except:
                obj_val += 1e10

        return obj_val

    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                         iterations: int = 100, epsilon: float = 1e-6) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Gradient Descent: θ_{t+1} = θ_t - η * ∇f(θ_t)
        """
        x = x0.copy().astype(float)
        path = [x.copy()]
        f_values = [self._objective_wrapper(x)]

        for iter_num in range(iterations):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                x_minus = x.copy()
                x_minus[i] -= epsilon

                if self.constraint_func is not None:
                    grad[i] = (self._objective_with_penalty(x_plus) -
                              self._objective_with_penalty(x_minus)) / (2 * epsilon)
                else:
                    grad[i] = (self._objective_wrapper(x_plus) -
                              self._objective_wrapper(x_minus)) / (2 * epsilon)

            x_new = x - learning_rate * grad

            for i, (lb, ub) in enumerate(self.bounds):
                x_new[i] = np.clip(x_new[i], lb, ub)

            x = x_new
            path.append(x.copy())
            f_values.append(self._objective_wrapper(x))

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break

        return x, path, f_values

    def optimize(self, method: str = 'scipy', learning_rate: float = 0.01,
                 iterations: int = 100, x0: np.ndarray = None) -> Dict:
        """Run optimization and return results."""
        if x0 is None:
            x0 = np.array([np.random.uniform(lb, ub) for lb, ub in self.bounds])

        f_values = []

        if method == 'gradient_descent':
            x_opt, path, f_values = self.gradient_descent(x0, learning_rate, iterations)
            f_opt = self._objective_wrapper(x_opt)
        else:
            path = [x0.copy()]
            f_values = [self._objective_wrapper(x0)]

            def callback(xk):
                path.append(xk.copy())
                f_values.append(self._objective_wrapper(xk))

            constraints = []
            if self.constraint_func is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: self.constraint_func(*x)
                })

            result = minimize(
                self._objective_wrapper,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints,
                callback=callback,
                options={'maxiter': iterations, 'ftol': 1e-9}
            )

            x_opt = result.x
            f_opt = result.fun

        self.optimization_path = path

        return {
            'x_optimal': x_opt,
            'f_optimal': f_opt,
            'path': path,
            'f_values': f_values,
            'start_point': x0,
            'learning_rate': learning_rate,
            'iterations': len(path) - 1
        }

    def find_maximum(self, **kwargs) -> Dict:
        """Find maximum by minimizing negative of function."""
        original_func = self.objective_func
        self.objective_func = lambda *args: -original_func(*args)
        result = self.optimize(**kwargs)
        self.objective_func = original_func
        result['f_optimal'] = -result['f_optimal']
        result['f_values'] = [-f for f in result['f_values']]
        return result
