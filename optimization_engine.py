"""
Optimization Engine
==================
Handle optimization with constraint support and path tracking.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Callable, Dict, List, Optional, Tuple


class OptimizationEngine:
    """Handle optimization with constraint support and path tracking."""

    def __init__(self, objective_func: Callable, constraint_func: Optional[Callable] = None,
                 num_vars: int = 2, bounds: List[Tuple[float, float]] = None,
                 constraint_type: str = 'eq'):
        self.objective_func = objective_func
        self.constraint_func = constraint_func
        self.num_vars = num_vars
        self.bounds = bounds or [(0.01, 10.0)] * num_vars
        self.optimization_path = []
        # constraint_type: 'eq' for =0, 'ineq_ge' for >=0, 'ineq_le' for <=0
        self.constraint_type = constraint_type

    def _objective_wrapper(self, x: np.ndarray) -> float:
        """Wrapper for objective function that handles arrays."""
        try:
            result = self.objective_func(*x)
            result = float(result)
            if np.isnan(result) or np.isinf(result):
                return 1e10
            return result
        except:
            return 1e10

    def _objective_with_penalty(self, x: np.ndarray, penalty_weight: float = 1000000) -> float:
        """Objective with constraint penalty."""
        obj_val = self._objective_wrapper(x)

        if self.constraint_func is not None:
            try:
                constraint_val = self.constraint_func(*x)
                # Use higher penalty for equality constraints
                penalty = penalty_weight * constraint_val ** 2
                obj_val += penalty
            except:
                obj_val += 1e10

        return obj_val

    def _find_constraint_point(self, direction: np.ndarray, search_range: Tuple[float, float] = (0.01, 10.0)) -> Optional[np.ndarray]:
        """Find a point on the constraint surface along a given direction."""
        if self.constraint_func is None:
            return None

        direction = direction / np.linalg.norm(direction)

        # Binary search to find point where constraint = 0
        low, high = search_range
        for _ in range(50):
            mid = (low + high) / 2
            point = mid * direction
            try:
                c_val = self.constraint_func(*point)
                if abs(c_val) < 1e-8:
                    return point
                # For constraints like a^2 + b^2 - 1 = 0, positive means outside circle
                if c_val > 0:
                    high = mid
                else:
                    low = mid
            except:
                high = mid

        return ((low + high) / 2) * direction

    def _grid_search_on_constraint(self, n_points: int = 360) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """Grid search on the constraint manifold for 2D problems."""
        best_x = None
        best_f = float('inf')
        path = []
        f_values = []

        if self.num_vars == 2:
            # For 2D, parameterize by angle and find radius that satisfies constraint
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                direction = np.array([np.cos(angle), np.sin(angle)])

                # Find the point on constraint surface
                point = self._find_constraint_point(direction)
                if point is None:
                    continue

                try:
                    # Check if constraint is satisfied
                    c_val = self.constraint_func(*point)
                    if abs(c_val) > 0.01:
                        continue

                    # Check if within bounds
                    valid = True
                    for j, (lb, ub) in enumerate(self.bounds):
                        if point[j] < lb or point[j] > ub:
                            valid = False
                            break

                    if not valid:
                        continue

                    f_val = self._objective_wrapper(point)
                    path.append(point.copy())
                    f_values.append(f_val)

                    if f_val < best_f:
                        best_f = f_val
                        best_x = point.copy()
                except:
                    continue

        return best_x, best_f, path, f_values

    def _constrained_optimize_1d(self) -> Dict:
        """
        For 2D problems with equality constraint, reduce to 1D optimization.
        Uses parametric search along the constraint manifold.
        """
        if self.num_vars != 2 or self.constraint_func is None:
            return None

        path = []
        f_values = []

        def objective_1d(theta):
            """Objective as function of angle parameter."""
            direction = np.array([np.cos(theta), np.sin(theta)])
            point = self._find_constraint_point(direction)

            if point is None:
                return 1e10

            try:
                c_val = self.constraint_func(*point)
                if abs(c_val) > 0.01:
                    return 1e10

                # Check bounds
                for j, (lb, ub) in enumerate(self.bounds):
                    if point[j] < lb - 1e-6 or point[j] > ub + 1e-6:
                        return 1e10

                f_val = self._objective_wrapper(point)
                path.append(point.copy())
                f_values.append(f_val)
                return f_val
            except:
                return 1e10

        # Use scipy to minimize over theta in [0, 2*pi]
        best_theta = None
        best_f = float('inf')

        # Grid search first
        for i in range(720):
            theta = 2 * np.pi * i / 720
            f = objective_1d(theta)
            if f < best_f:
                best_f = f
                best_theta = theta

        # Refine with Brent's method
        if best_theta is not None:
            try:
                result = minimize_scalar(
                    objective_1d,
                    bounds=(best_theta - np.pi/36, best_theta + np.pi/36),
                    method='bounded'
                )
                if result.fun < best_f:
                    best_theta = result.x
                    best_f = result.fun
            except:
                pass

        if best_theta is None:
            return None

        # Get the optimal point
        direction = np.array([np.cos(best_theta), np.sin(best_theta)])
        x_opt = self._find_constraint_point(direction)

        if x_opt is None:
            return None

        return {
            'x_optimal': x_opt,
            'f_optimal': best_f,
            'path': path,
            'f_values': f_values
        }

    def _project_to_constraint(self, x: np.ndarray, prev_x: np.ndarray = None, max_angle_step: float = None) -> np.ndarray:
        """
        Project point back onto constraint surface (for 2D equality constraints).

        Args:
            x: Target point to project
            prev_x: Previous point on constraint (for limiting step size)
            max_angle_step: Maximum angular step in radians (for gradual convergence)
        """
        if self.constraint_func is None or self.num_vars != 2:
            return x

        # Use the direction from origin to find point on constraint
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return x

        direction = x / norm

        # If we have a previous point and max_angle_step, limit the angular change
        if prev_x is not None and max_angle_step is not None:
            prev_norm = np.linalg.norm(prev_x)
            if prev_norm > 1e-10:
                prev_direction = prev_x / prev_norm

                # Calculate current angle and previous angle
                current_angle = np.arctan2(direction[1], direction[0])
                prev_angle = np.arctan2(prev_direction[1], prev_direction[0])

                # Calculate angular difference (handle wraparound)
                angle_diff = current_angle - prev_angle
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                # Limit the angular step
                if abs(angle_diff) > max_angle_step:
                    # Clamp to max_angle_step
                    clamped_diff = np.sign(angle_diff) * max_angle_step
                    new_angle = prev_angle + clamped_diff
                    direction = np.array([np.cos(new_angle), np.sin(new_angle)])

        projected = self._find_constraint_point(direction)

        if projected is not None:
            # Verify projection is valid
            try:
                c_val = abs(self.constraint_func(*projected))
                if c_val < 0.1:
                    return projected
            except:
                pass

        return x

    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                         iterations: int = 100, epsilon: float = 1e-6) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Gradient Descent with momentum and adaptive learning rate for constrained problems.
        For constrained 2D problems, uses projection with limited angular steps for smooth convergence.
        """
        x = x0.copy().astype(float)
        path = [x.copy()]
        current_f = self._objective_wrapper(x)
        f_values = [current_f]

        # Minimum distance from zero to avoid singularities like 1/x
        min_dist_from_zero = 0.01

        best_x = x.copy()
        best_f = current_f

        # For constrained 2D problems, use angular step limiting for gradual convergence
        # This creates a smooth path along the constraint for visualization
        is_constrained_2d = (self.constraint_func is not None and
                            self.constraint_type == 'eq' and
                            self.num_vars == 2)

        if is_constrained_2d:
            # Use angular stepping along constraint for smooth visualization
            # Max angular step: ~1 degree per iteration for smooth path
            max_angle_step = np.radians(1.0)  # 1 degree per step

            for iter_num in range(iterations):
                # Compute gradient of objective function
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    grad[i] = (self._objective_wrapper(x_plus) -
                              self._objective_wrapper(x_minus)) / (2 * epsilon)

                # Clip gradient
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 100:
                    grad = grad / grad_norm * 100

                # Take gradient step
                x_new = x - learning_rate * grad

                # Project back to constraint with limited angular step
                x_new = self._project_to_constraint(x_new, prev_x=x, max_angle_step=max_angle_step)

                # Apply singularity protection
                for i in range(len(x_new)):
                    if abs(x_new[i]) < min_dist_from_zero:
                        x_new[i] = min_dist_from_zero if x_new[i] >= 0 else -min_dist_from_zero

                # Evaluate new point
                new_f = self._objective_wrapper(x_new)

                if new_f < 1e9 and not np.isnan(new_f) and not np.isinf(new_f):
                    x = x_new
                    current_f = new_f

                    if current_f < best_f:
                        best_f = current_f
                        best_x = x.copy()

                path.append(x.copy())
                f_values.append(current_f)

                # Check convergence (angular change very small)
                if grad_norm < 1e-6:
                    break
        else:
            # Standard gradient descent for unconstrained or non-2D problems
            momentum = 0.9
            velocity = np.zeros_like(x)
            adaptive_lr = learning_rate

            for iter_num in range(iterations):
                # Compute gradient
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    grad[i] = (self._objective_wrapper(x_plus) -
                              self._objective_wrapper(x_minus)) / (2 * epsilon)

                # Clip gradient
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 100:
                    grad = grad / grad_norm * 100

                # Update with momentum
                velocity = momentum * velocity - adaptive_lr * grad
                x_new = x + velocity

                # Apply bounds
                for i, (lb, ub) in enumerate(self.bounds):
                    x_new[i] = np.clip(x_new[i], lb, ub)

                # Singularity protection
                for i in range(len(x_new)):
                    if abs(x_new[i]) < min_dist_from_zero:
                        x_new[i] = min_dist_from_zero if x_new[i] >= 0 else -min_dist_from_zero

                # Evaluate
                new_f = self._objective_wrapper(x_new)

                if new_f < 1e9 and not np.isnan(new_f) and not np.isinf(new_f):
                    if new_f > current_f * 1.1:
                        adaptive_lr *= 0.7
                        velocity *= 0.5
                    elif new_f < current_f * 0.99:
                        adaptive_lr = min(adaptive_lr * 1.05, learning_rate)

                    x = x_new
                    current_f = new_f

                    if current_f < best_f:
                        best_f = current_f
                        best_x = x.copy()
                else:
                    adaptive_lr *= 0.5
                    velocity *= 0.3

                path.append(x.copy())
                f_values.append(current_f)

                if grad_norm < 1e-6 or adaptive_lr < 1e-8:
                    break

        return best_x, path, f_values

    def _generate_safe_starting_point(self) -> np.ndarray:
        """Generate a starting point that avoids potential singularities (like 1/x near 0)."""
        # For 2D constrained problems, try to find a good starting point on the constraint
        if self.num_vars == 2 and self.constraint_func is not None:
            # Try multiple angles to find a point on constraint where objective is finite
            # Start with angles far from 45 degrees (likely optimal for symmetric problems)
            # This gives better visualization of the optimization path with gradual convergence
            best_x0 = None
            best_f = float('inf')
            # Start at angles like 10, 15, 20 degrees for longer path to 45° optimal
            for angle_deg in [10, 15, 20, 70, 75, 80, 100, 110, 160, 170, 190, 200, 250, 260, 280, 290, 340, 350]:
                angle = np.radians(angle_deg)
                direction = np.array([np.cos(angle), np.sin(angle)])
                point = self._find_constraint_point(direction)
                if point is not None:
                    try:
                        c_val = abs(self.constraint_func(*point))
                        f_val = self._objective_wrapper(point)
                        # Check if this point has a finite objective value
                        if c_val < 0.1 and f_val < 1e9 and not np.isnan(f_val) and not np.isinf(f_val):
                            if f_val < best_f:
                                best_f = f_val
                                best_x0 = point.copy()
                    except:
                        continue
            if best_x0 is not None:
                return best_x0

        # Fallback: generate point based on bounds
        x0 = []
        for lb, ub in self.bounds:
            # Always try to start at a value away from zero (like 1.0) if it's within bounds
            # This helps with functions that have 1/x terms
            if lb <= 1.0 <= ub:
                # 1.0 is within bounds - use a value around 1.0
                x0.append(np.random.uniform(0.8, min(1.5, ub)))
            elif lb > 0:
                # All positive - start in the middle, but not too close to zero
                mid = (lb + ub) / 2
                safe_val = max(mid, lb + 0.1)
                x0.append(safe_val)
            elif ub < 0:
                # All negative - start in the middle, but not too close to zero
                mid = (lb + ub) / 2
                safe_val = min(mid, ub - 0.1)
                x0.append(safe_val)
            else:
                # Mixed bounds (lb < 0 < ub) - start at positive value away from zero
                x0.append(np.random.uniform(0.5, max(2.0, ub)))
        return np.array(x0)

    def optimize(self, method: str = 'scipy', learning_rate: float = 0.01,
                 iterations: int = 100, x0: np.ndarray = None) -> Dict:
        """Run optimization and return results."""
        if x0 is None:
            x0 = self._generate_safe_starting_point()

        f_values = []

        if method == 'gradient_descent':
            x_opt, path, f_values = self.gradient_descent(x0, learning_rate, iterations)
            f_opt = self._objective_wrapper(x_opt)
        else:
            # For 2D equality constraints, use parametric/grid search approach first
            # This is more reliable than scipy constrained optimization
            if self.constraint_func is not None and self.constraint_type == 'eq' and self.num_vars == 2:
                result_1d = self._constrained_optimize_1d()
                if result_1d is not None and result_1d['f_optimal'] < 1e9:
                    # Verify constraint satisfaction
                    c_val = abs(self.constraint_func(*result_1d['x_optimal']))
                    if c_val < 0.01:
                        self.optimization_path = result_1d['path']
                        return {
                            'x_optimal': result_1d['x_optimal'],
                            'f_optimal': result_1d['f_optimal'],
                            'path': result_1d['path'],
                            'f_values': result_1d['f_values'],
                            'start_point': x0,
                            'learning_rate': learning_rate,
                            'iterations': len(result_1d['path']) - 1
                        }

            path = [x0.copy()]
            f_values = [self._objective_wrapper(x0)]

            def callback(xk):
                path.append(xk.copy())
                f_values.append(self._objective_wrapper(xk))

            constraints = []
            if self.constraint_func is not None:
                # Capture constraint_func in closure properly
                cfunc = self.constraint_func
                if self.constraint_type == 'eq':
                    # g(x) = 0
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, f=cfunc: f(*x)
                    })
                elif self.constraint_type == 'ineq_ge':
                    # g(x) >= 0 (scipy 'ineq' means fun(x) >= 0)
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, f=cfunc: f(*x)
                    })
                elif self.constraint_type == 'ineq_le':
                    # g(x) <= 0, equivalent to -g(x) >= 0
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, f=cfunc: -f(*x)
                    })

            best_x = None
            best_f = float('inf')
            best_path = path
            best_f_values = f_values

            # Try SLSQP with the given starting point (with relaxed bounds for constrained problems)
            try:
                # For equality constraints, allow negative values
                opt_bounds = self.bounds
                if self.constraint_func is not None and self.constraint_type == 'eq':
                    opt_bounds = [(-10.0, 10.0)] * self.num_vars

                result = minimize(
                    self._objective_wrapper,
                    x0,
                    method='SLSQP',
                    bounds=opt_bounds,
                    constraints=constraints,
                    callback=callback,
                    options={'maxiter': iterations, 'ftol': 1e-9}
                )
                if not np.isnan(result.fun) and result.fun < best_f:
                    # Verify constraint is satisfied
                    if self.constraint_func is None or abs(self.constraint_func(*result.x)) < 0.01:
                        best_x, best_f = result.x.copy(), result.fun
                        best_path, best_f_values = path.copy(), f_values.copy()
            except:
                pass

            # Try trust-constr method (more robust for nonlinear constraints)
            if self.constraint_func and self.constraint_type == 'eq':
                try:
                    from scipy.optimize import NonlinearConstraint
                    path2 = [x0.copy()]
                    f_values2 = [self._objective_wrapper(x0)]

                    def callback2(xk, state=None):
                        path2.append(xk.copy())
                        f_values2.append(self._objective_wrapper(xk))
                        return False

                    cfunc = self.constraint_func
                    nlc = NonlinearConstraint(lambda x: cfunc(*x), -1e-6, 1e-6)
                    opt_bounds = [(-10.0, 10.0)] * self.num_vars
                    result2 = minimize(
                        self._objective_wrapper,
                        x0,
                        method='trust-constr',
                        bounds=opt_bounds,
                        constraints=nlc,
                        callback=callback2,
                        options={'maxiter': iterations}
                    )
                    if not np.isnan(result2.fun) and result2.fun < best_f:
                        if abs(self.constraint_func(*result2.x)) < 0.01:
                            best_x, best_f = result2.x.copy(), result2.fun
                            best_path, best_f_values = path2.copy(), f_values2.copy()
                except:
                    pass

            # Try multiple random starting points with SLSQP
            if best_x is None or best_f >= 1e9:
                for _ in range(10):
                    try:
                        # Generate starting point avoiding zero (to handle 1/x singularities)
                        if self.num_vars == 2:
                            # Use positive random values away from zero
                            test_x0 = np.array([
                                np.random.uniform(0.5, 3.0),
                                np.random.uniform(0.5, 3.0)
                            ])
                        else:
                            # Use safe starting points away from zero
                            test_x0 = np.array([np.random.uniform(0.5, 3.0) for _ in range(self.num_vars)])

                        result3 = minimize(
                            self._objective_wrapper,
                            test_x0,
                            method='SLSQP',
                            bounds=self.bounds,
                            constraints=constraints,
                            options={'maxiter': iterations, 'ftol': 1e-9}
                        )
                        if not np.isnan(result3.fun) and result3.fun < best_f:
                            if self.constraint_func is None or abs(self.constraint_func(*result3.x)) < 0.01:
                                best_x, best_f = result3.x.copy(), result3.fun
                    except:
                        pass

            # Final fallback: use penalty method with gradient descent
            if best_x is None or best_f >= 1e9:
                x_opt, path, f_values = self.gradient_descent(x0, 0.01, iterations)
                f_opt = self._objective_wrapper(x_opt)
                best_x, best_f = x_opt, f_opt
                best_path, best_f_values = path, f_values

            x_opt = best_x if best_x is not None else x0
            f_opt = best_f if best_f < float('inf') else self._objective_wrapper(x0)
            path = best_path
            f_values = best_f_values

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
