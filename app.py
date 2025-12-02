"""
Inequality Explorer - Interactive Min/Max Optimization Tool
===========================================================
An educational tool for exploring minimum/maximum of multi-variable functions
under constraints, simulating the process of finding "Equality Cases" (Điểm rơi).

Author: Generated with Claude Code
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, NonlinearConstraint
from typing import Callable, Tuple, List, Dict, Optional
import re
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Inequality Explorer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe mathematical functions allowed in expressions
SAFE_MATH_FUNCTIONS = {
    'sqrt': np.sqrt,
    'abs': np.abs,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,
    'log10': np.log10,
    'log2': np.log2,
    'pi': np.pi,
    'e': np.e,
    'pow': np.power,
    'min': np.minimum,
    'max': np.maximum,
}


# =============================================================================
# SAFE EXPRESSION PARSER
# =============================================================================

class SafeExpressionParser:
    """Safely parse and evaluate mathematical expressions."""

    # Allowed tokens pattern
    ALLOWED_PATTERN = re.compile(
        r'^[\d\s\+\-\*\/\(\)\.\,\^a-zA-Z_]+$'
    )

    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        r'__',           # Dunder methods
        r'import',       # Import statements
        r'exec',         # Exec function
        r'eval',         # Nested eval
        r'open',         # File operations
        r'os\.',         # OS module
        r'sys\.',        # Sys module
        r'subprocess',   # Subprocess
        r'compile',      # Compile function
    ]

    @classmethod
    def validate_expression(cls, expr: str) -> Tuple[bool, str]:
        """Validate if expression is safe to evaluate."""
        if not expr or not expr.strip():
            return False, "Expression cannot be empty"

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}"

        return True, "Valid"

    @classmethod
    def prepare_expression(cls, expr: str) -> str:
        """Prepare expression for evaluation (handle ^ for power)."""
        # Replace ^ with ** for power operation
        expr = expr.replace('^', '**')
        return expr

    @classmethod
    def create_function(cls, expr: str, variables: List[str]) -> Callable:
        """Create a callable function from expression string."""
        is_valid, msg = cls.validate_expression(expr)
        if not is_valid:
            raise ValueError(msg)

        prepared_expr = cls.prepare_expression(expr)

        def func(*args):
            if len(args) != len(variables):
                raise ValueError(f"Expected {len(variables)} arguments, got {len(args)}")

            # Create namespace with safe functions and variables
            namespace = SAFE_MATH_FUNCTIONS.copy()
            for var, val in zip(variables, args):
                namespace[var] = val

            try:
                return eval(prepared_expr, {"__builtins__": {}}, namespace)
            except Exception as e:
                raise ValueError(f"Error evaluating expression: {e}")

        return func


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================

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
            return 1e10  # Return large value for invalid points

    def _objective_with_penalty(self, x: np.ndarray, penalty_weight: float = 1000) -> float:
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
                         iterations: int = 100, epsilon: float = 1e-7) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """Manual gradient descent with path tracking and function values."""
        x = x0.copy().astype(float)
        path = [x.copy()]
        f_values = [self._objective_wrapper(x)]

        for iter_num in range(iterations):
            # Numerical gradient
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                x_minus = x.copy()
                x_minus[i] -= epsilon
                grad[i] = (self._objective_with_penalty(x_plus) -
                          self._objective_with_penalty(x_minus)) / (2 * epsilon)

            # Update step with gradient descent
            x = x - learning_rate * grad

            # Clip to bounds
            for i, (lb, ub) in enumerate(self.bounds):
                x[i] = np.clip(x[i], lb, ub)

            path.append(x.copy())
            f_values.append(self._objective_wrapper(x))

            # Check convergence
            if np.linalg.norm(grad) < epsilon:
                break

        return x, path, f_values

    def optimize(self, method: str = 'scipy', learning_rate: float = 0.01,
                 iterations: int = 100, x0: np.ndarray = None) -> Dict:
        """Run optimization and return results."""
        if x0 is None:
            # Random starting point within bounds
            x0 = np.array([np.random.uniform(lb, ub) for lb, ub in self.bounds])

        f_values = []

        if method == 'gradient_descent':
            x_opt, path, f_values = self.gradient_descent(x0, learning_rate, iterations)
            f_opt = self._objective_wrapper(x_opt)
        else:
            # Use scipy.optimize.minimize with callback
            path = [x0.copy()]
            f_values = [self._objective_wrapper(x0)]

            def callback(xk):
                path.append(xk.copy())
                f_values.append(self._objective_wrapper(xk))

            # Setup constraints if provided
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


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class Visualizer:
    """Create interactive visualizations for optimization."""

    @staticmethod
    def create_3d_surface(func: Callable, x_range: Tuple[float, float],
                          y_range: Tuple[float, float], resolution: int = 50,
                          var_names: List[str] = ['a', 'b']) -> go.Figure:
        """Create 3D surface plot."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = func(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan

        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.8,
            name='f(a,b)'
        )])

        fig.update_layout(
            title='3D Surface Plot of f({}, {})'.format(*var_names),
            scene=dict(
                xaxis_title=var_names[0],
                yaxis_title=var_names[1],
                zaxis_title='f({}, {})'.format(*var_names),
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    @staticmethod
    def add_optimization_path_3d(fig: go.Figure, path: List[np.ndarray],
                                  func: Callable, color: str = 'red') -> go.Figure:
        """Add optimization path to 3D surface plot with enhanced visualization."""
        if len(path) < 2:
            return fig

        path_array = np.array(path)
        z_path = []
        for p in path:
            try:
                z_path.append(func(p[0], p[1]))
            except:
                z_path.append(np.nan)

        # Add path line with gradient coloring
        fig.add_trace(go.Scatter3d(
            x=path_array[:, 0],
            y=path_array[:, 1],
            z=z_path,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(
                size=4,
                color=list(range(len(path))),
                colorscale='Reds',
                showscale=False
            ),
            name='Optimization Path'
        ))

        # Highlight start point (green circle)
        fig.add_trace(go.Scatter3d(
            x=[path_array[0, 0]],
            y=[path_array[0, 1]],
            z=[z_path[0]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='Start Point'
        ))

        # Highlight end point (gold star-like marker)
        fig.add_trace(go.Scatter3d(
            x=[path_array[-1, 0]],
            y=[path_array[-1, 1]],
            z=[z_path[-1]],
            mode='markers',
            marker=dict(size=14, color='gold', symbol='diamond',
                       line=dict(color='black', width=2)),
            name='Optimal Point ★'
        ))

        return fig

    @staticmethod
    def create_contour_with_trajectory(func: Callable, x_range: Tuple[float, float],
                                        y_range: Tuple[float, float],
                                        path: List[np.ndarray] = None,
                                        resolution: int = 100,
                                        var_names: List[str] = ['a', 'b'],
                                        constraint_func: Callable = None,
                                        show_arrows: bool = True) -> go.Figure:
        """Create contour plot with full trajectory visualization including arrows."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = func(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan

        fig = go.Figure()

        # Add contour as background
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            name='f({}, {})'.format(*var_names),
            opacity=0.9,
            colorbar=dict(title='f(a,b)', x=1.02)
        ))

        # Add constraint curve if provided
        if constraint_func is not None:
            C = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        C[i, j] = constraint_func(X[i, j], Y[i, j])
                    except:
                        C[i, j] = np.nan

            fig.add_trace(go.Contour(
                x=x, y=y, z=C,
                contours=dict(
                    start=0, end=0, size=0.01,
                    coloring='lines'
                ),
                line=dict(color='red', width=4, dash='dash'),
                showscale=False,
                name='Constraint (g=0)'
            ))

        # Add optimization path with arrows
        if path is not None and len(path) >= 2:
            path_array = np.array(path)
            n_points = len(path_array)

            # Main trajectory line with gradient coloring
            for i in range(n_points - 1):
                # Color gradient from blue (start) to red (end)
                t = i / max(n_points - 2, 1)
                color = f'rgb({int(255*t)}, {int(100*(1-t))}, {int(255*(1-t))})'

                fig.add_trace(go.Scatter(
                    x=[path_array[i, 0], path_array[i+1, 0]],
                    y=[path_array[i, 1], path_array[i+1, 1]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add arrows to show direction (every few points)
            if show_arrows:
                arrow_interval = max(1, n_points // 10)  # Show ~10 arrows
                for i in range(0, n_points - 1, arrow_interval):
                    dx = path_array[i+1, 0] - path_array[i, 0]
                    dy = path_array[i+1, 1] - path_array[i, 1]
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0.001:  # Only add arrow if there's movement
                        # Normalize and scale arrow
                        scale = min(0.15, length * 0.5)
                        dx_norm = dx / length * scale
                        dy_norm = dy / length * scale

                        # Arrow position (middle of segment)
                        ax = path_array[i, 0] + dx * 0.5
                        ay = path_array[i, 1] + dy * 0.5

                        fig.add_annotation(
                            x=ax + dx_norm,
                            y=ay + dy_norm,
                            ax=ax,
                            ay=ay,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=2,
                            arrowcolor='rgba(255, 100, 100, 0.8)'
                        )

            # Path points with size gradient
            sizes = np.linspace(6, 3, n_points)
            fig.add_trace(go.Scatter(
                x=path_array[:, 0],
                y=path_array[:, 1],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=list(range(n_points)),
                    colorscale='RdYlBu_r',
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                name='Path Points',
                hovertemplate='Step %{marker.color}<br>a=%{x:.4f}<br>b=%{y:.4f}<extra></extra>'
            ))

            # Start point - Green circle with label
            fig.add_trace(go.Scatter(
                x=[path_array[0, 0]],
                y=[path_array[0, 1]],
                mode='markers+text',
                marker=dict(size=18, color='limegreen', symbol='circle',
                           line=dict(color='darkgreen', width=3)),
                text=['START'],
                textposition='top center',
                textfont=dict(size=12, color='darkgreen', family='Arial Black'),
                name='Start Point'
            ))

            # End point - Gold star with label
            fig.add_trace(go.Scatter(
                x=[path_array[-1, 0]],
                y=[path_array[-1, 1]],
                mode='markers+text',
                marker=dict(size=22, color='gold', symbol='star',
                           line=dict(color='darkorange', width=3)),
                text=['MIN ★'],
                textposition='bottom center',
                textfont=dict(size=14, color='darkorange', family='Arial Black'),
                name='Optimal Point ★'
            ))

        fig.update_layout(
            title=dict(
                text='Optimization Trajectory on Contour Plot',
                font=dict(size=16)
            ),
            xaxis_title=var_names[0],
            yaxis_title=var_names[1],
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            hovermode='closest'
        )

        return fig

    @staticmethod
    def create_convergence_plot(f_values: List[float], learning_rate: float) -> go.Figure:
        """Create a plot showing convergence of function value over iterations."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(f_values))),
            y=f_values,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            name='f(a,b)'
        ))

        # Add horizontal line at final value
        fig.add_hline(
            y=f_values[-1],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Final: {f_values[-1]:.4f}"
        )

        fig.update_layout(
            title=f'Convergence Plot (LR={learning_rate})',
            xaxis_title='Iteration',
            yaxis_title='f(a,b)',
            height=300,
            showlegend=True
        )

        return fig

    @staticmethod
    def create_contour_plot(func: Callable, x_range: Tuple[float, float],
                            y_range: Tuple[float, float], resolution: int = 100,
                            var_names: List[str] = ['a', 'b'],
                            constraint_func: Callable = None) -> go.Figure:
        """Create contour plot with optional constraint curve (legacy method)."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = func(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan

        fig = go.Figure()

        # Add contour
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            name='f({}, {})'.format(*var_names)
        ))

        # Add constraint curve if provided
        if constraint_func is not None:
            C = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        C[i, j] = constraint_func(X[i, j], Y[i, j])
                    except:
                        C[i, j] = np.nan

            fig.add_trace(go.Contour(
                x=x, y=y, z=C,
                contours=dict(
                    start=0, end=0, size=0.01,
                    coloring='lines'
                ),
                line=dict(color='red', width=3),
                showscale=False,
                name='Constraint'
            ))

        fig.update_layout(
            title='Contour Plot of f({}, {})'.format(*var_names),
            xaxis_title=var_names[0],
            yaxis_title=var_names[1],
            height=500,
        )

        return fig

    @staticmethod
    def add_path_to_contour(fig: go.Figure, path: List[np.ndarray],
                            color: str = 'red') -> go.Figure:
        """Add optimization path to contour plot (legacy method)."""
        if len(path) < 2:
            return fig

        path_array = np.array(path)

        # Add path line
        fig.add_trace(go.Scatter(
            x=path_array[:, 0],
            y=path_array[:, 1],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            name='Optimization Path'
        ))

        # Start point
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]],
            y=[path_array[0, 1]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='diamond'),
            name='Start'
        ))

        # End point
        fig.add_trace(go.Scatter(
            x=[path_array[-1, 0]],
            y=[path_array[-1, 1]],
            mode='markers',
            marker=dict(size=12, color='blue', symbol='x'),
            name='Optimal'
        ))

        return fig


# =============================================================================
# MATH INSIGHT ANALYZER
# =============================================================================

class MathInsightAnalyzer:
    """Analyze optimization results for mathematical insights."""

    TOLERANCE = 0.05  # Relative tolerance for comparisons

    @classmethod
    def analyze(cls, x_optimal: np.ndarray, var_names: List[str],
                f_optimal: float, constraint_expr: str = None) -> List[Dict]:
        """Analyze the optimal point and generate insights."""
        insights = []

        # Check for symmetry
        symmetry_insight = cls._check_symmetry(x_optimal, var_names)
        if symmetry_insight:
            insights.append(symmetry_insight)

        # Check for boundary conditions
        boundary_insight = cls._check_boundary(x_optimal, var_names)
        if boundary_insight:
            insights.append(boundary_insight)

        # Check for specific ratios
        ratio_insights = cls._check_ratios(x_optimal, var_names)
        insights.extend(ratio_insights)

        # Check for special values
        special_insights = cls._check_special_values(x_optimal, var_names)
        insights.extend(special_insights)

        # Add general suggestion based on structure
        general_insight = cls._generate_general_suggestion(x_optimal, var_names, f_optimal)
        if general_insight:
            insights.append(general_insight)

        return insights

    @classmethod
    def _check_symmetry(cls, x: np.ndarray, var_names: List[str]) -> Optional[Dict]:
        """Check if variables are approximately equal (symmetric case)."""
        if len(x) < 2:
            return None

        mean_val = np.mean(x)
        if mean_val == 0:
            return None

        # Check if all values are close to the mean
        relative_diffs = np.abs(x - mean_val) / abs(mean_val)

        if np.all(relative_diffs < cls.TOLERANCE):
            vars_str = ' = '.join(var_names)
            return {
                'type': 'symmetry',
                'icon': '🔄',
                'title': 'Symmetry Detected',
                'description': f'All variables are approximately equal: {vars_str} ≈ {mean_val:.4f}',
                'suggestion': 'This suggests using AM-GM inequality or Cauchy-Schwarz at the symmetric point. '
                             'The equality case occurs when all variables are equal.',
                'technique': 'AM-GM / Cauchy-Schwarz'
            }

        # Check pairwise equality
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if abs(x[i]) > cls.TOLERANCE and abs(x[i] - x[j]) / abs(x[i]) < cls.TOLERANCE:
                    return {
                        'type': 'partial_symmetry',
                        'icon': '↔️',
                        'title': 'Partial Symmetry',
                        'description': f'{var_names[i]} ≈ {var_names[j]} ≈ {x[i]:.4f}',
                        'suggestion': f'Consider substituting {var_names[j]} = {var_names[i]} to reduce variables.',
                        'technique': 'Variable Substitution'
                    }

        return None

    @classmethod
    def _check_boundary(cls, x: np.ndarray, var_names: List[str]) -> Optional[Dict]:
        """Check if any variable is at or near boundary (0 or very small/large)."""
        boundary_vars = []

        for i, val in enumerate(x):
            if abs(val) < 0.1:  # Near zero
                boundary_vars.append((var_names[i], val, 'near zero'))
            elif val > 100:  # Very large
                boundary_vars.append((var_names[i], val, 'very large'))

        if boundary_vars:
            desc_parts = [f'{v[0]} → {v[2]} ({v[1]:.4f})' for v in boundary_vars]
            return {
                'type': 'boundary',
                'icon': '📍',
                'title': 'Boundary Case',
                'description': 'The minimum occurs at a boundary: ' + ', '.join(desc_parts),
                'suggestion': 'Consider analyzing the limiting behavior as variables approach 0 or infinity. '
                             'Use L\'Hôpital\'s rule or asymptotic analysis.',
                'technique': 'Boundary Analysis / L\'Hôpital'
            }

        return None

    @classmethod
    def _check_ratios(cls, x: np.ndarray, var_names: List[str]) -> List[Dict]:
        """Check for specific ratios between variables."""
        insights = []

        common_ratios = [
            (2, '2:1'),
            (3, '3:1'),
            (0.5, '1:2'),
            (1/3, '1:3'),
            (np.sqrt(2), '√2:1'),
            (np.sqrt(3), '√3:1'),
            ((1 + np.sqrt(5)) / 2, 'φ:1 (Golden Ratio)'),
        ]

        for i in range(len(x)):
            for j in range(len(x)):
                if i >= j or abs(x[j]) < 0.01:
                    continue

                ratio = x[i] / x[j]

                for target_ratio, ratio_name in common_ratios:
                    if abs(ratio - target_ratio) / target_ratio < cls.TOLERANCE:
                        insights.append({
                            'type': 'ratio',
                            'icon': '📐',
                            'title': f'Special Ratio: {ratio_name}',
                            'description': f'{var_names[i]} / {var_names[j]} ≈ {ratio:.4f} ({ratio_name})',
                            'suggestion': f'Try substituting {var_names[i]} = {target_ratio:.4f} × {var_names[j]} '
                                         f'to simplify the problem.',
                            'technique': 'Variable Ratio Substitution'
                        })
                        break

        return insights

    @classmethod
    def _check_special_values(cls, x: np.ndarray, var_names: List[str]) -> List[Dict]:
        """Check for special mathematical values."""
        insights = []

        special_values = [
            (1, '1'),
            (2, '2'),
            (np.e, 'e'),
            (np.pi, 'π'),
            (np.sqrt(2), '√2'),
            (np.sqrt(3), '√3'),
            ((1 + np.sqrt(5)) / 2, 'φ (Golden Ratio)'),
        ]

        for i, val in enumerate(x):
            for target, name in special_values:
                if abs(val - target) < cls.TOLERANCE * target:
                    insights.append({
                        'type': 'special_value',
                        'icon': '✨',
                        'title': f'Special Value: {name}',
                        'description': f'{var_names[i]} ≈ {name} = {target:.4f}',
                        'suggestion': f'The optimal value {var_names[i]} = {name} suggests this might be '
                                     f'derivable analytically using calculus or algebraic manipulation.',
                        'technique': 'Analytical Solution'
                    })
                    break

        return insights

    @classmethod
    def _generate_general_suggestion(cls, x: np.ndarray, var_names: List[str],
                                     f_optimal: float) -> Optional[Dict]:
        """Generate general suggestion based on the optimal point."""
        # Check if optimal value is a nice number
        nice_values = [0, 1, 2, 3, 4, 5, 0.5, 0.25, 0.75]

        for nice in nice_values:
            if abs(f_optimal - nice) < cls.TOLERANCE:
                return {
                    'type': 'nice_optimal',
                    'icon': '🎯',
                    'title': 'Clean Optimal Value',
                    'description': f'The optimal value f* ≈ {nice} is a "nice" number.',
                    'suggestion': 'This clean value often indicates the inequality can be proven '
                                 'elegantly using classical techniques like AM-GM, Cauchy-Schwarz, or Jensen.',
                    'technique': 'Classical Inequalities'
                }

        return None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Initialize session state for random seed
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = np.random.randint(0, 10000)

    st.title("📐 Inequality Explorer")
    st.markdown("""
    *An interactive tool for exploring minimum/maximum of multi-variable functions
    and discovering "Equality Cases" (Điểm rơi) in inequality problems.*
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Number of variables
    num_vars = st.sidebar.radio(
        "Number of Variables",
        options=[2, 3],
        index=0,
        help="Choose 2 or 3 variables for your function"
    )

    var_names = ['a', 'b', 'c'][:num_vars]

    # Optimization settings
    st.sidebar.subheader("Optimization Settings")

    optimization_method = st.sidebar.selectbox(
        "Method",
        options=['gradient_descent', 'scipy'],
        format_func=lambda x: 'Gradient Descent (Visual)' if x == 'gradient_descent' else 'SciPy (SLSQP)',
        help="Gradient Descent shows detailed path; SciPy is faster but fewer path points"
    )

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format="%.3f",
        help="Step size for optimization. Higher = faster but may overshoot/zigzag. Lower = smoother path but slower."
    )

    iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        help="Maximum number of optimization iterations"
    )

    # Variable bounds
    st.sidebar.subheader("Variable Bounds")
    bounds = []
    for var in var_names:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lb = st.number_input(f"{var} min", value=0.01, step=0.1, key=f"{var}_min")
        with col2:
            ub = st.number_input(f"{var} max", value=5.0, step=0.1, key=f"{var}_max")
        bounds.append((lb, ub))

    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_arrows = st.sidebar.checkbox("Show Direction Arrows", value=True,
                                      help="Display arrows showing descent direction on contour plot")
    show_convergence = st.sidebar.checkbox("Show Convergence Plot", value=True,
                                           help="Display how f(a,b) changes over iterations")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Function Input")

        # Example expressions
        examples = {
            '2 vars': {
                'Simple Sum': ('a**2 + b**2', 'a + b - 2'),
                'AM-GM Example': ('a + b + 1/a + 1/b', 'a*b - 1'),
                'Cauchy-Schwarz': ('(a**2 + b**2) * (1/a**2 + 1/b**2)', ''),
                'Rosenbrock': ('(1-a)**2 + 100*(b-a**2)**2', ''),
                'Custom': ('', ''),
            },
            '3 vars': {
                'Symmetric': ('a**2 + b**2 + c**2', 'a + b + c - 3'),
                'AM-GM 3 vars': ('a + b + c + 1/a + 1/b + 1/c', 'a*b*c - 1'),
                'Mixed': ('a**2 + b**2 + c**2 + 2/(a*b*c)', 'a + b + c - 3'),
                'Custom': ('', ''),
            }
        }

        example_key = '2 vars' if num_vars == 2 else '3 vars'
        selected_example = st.selectbox(
            "Load Example",
            options=list(examples[example_key].keys()),
            index=0
        )

        default_expr, default_constraint = examples[example_key][selected_example]

        # Function expression input
        func_expr = st.text_input(
            f"f({', '.join(var_names)}) =",
            value=default_expr,
            placeholder=f"e.g., a**2 + b**2 + 1/a + 1/b",
            help="Enter mathematical expression. Use ** for power, sqrt(), sin(), cos(), exp(), log()"
        )

        # Constraint input
        constraint_expr = st.text_input(
            "Constraint (= 0)",
            value=default_constraint,
            placeholder="e.g., a + b - 2 (means a + b = 2)",
            help="Enter constraint equation. Leave empty for no constraint."
        )

        # Find min or max
        find_type = st.radio(
            "Find",
            options=['Minimum', 'Maximum'],
            horizontal=True
        )

    with col2:
        st.subheader("📊 Starting Point")

        # Starting point options
        start_mode = st.radio(
            "Starting Point Mode",
            options=['Random', 'Custom'],
            horizontal=True,
            help="Choose random or specify custom starting point"
        )

        if start_mode == 'Custom':
            st.write("Specify starting coordinates:")
            start_cols = st.columns(num_vars)
            start_point = []
            for i, var in enumerate(var_names):
                with start_cols[i]:
                    val = st.number_input(f"{var}₀", value=2.0, step=0.1, key=f"start_{var}")
                    start_point.append(val)
            start_point = np.array(start_point)
        else:
            start_point = None
            # Randomize button
            if st.button("🎲 Randomize Start Point", help="Generate new random starting point"):
                st.session_state.random_seed = np.random.randint(0, 10000)
                st.rerun()

            st.caption(f"Current seed: {st.session_state.random_seed}")

        st.subheader("📈 Plot Settings")
        resolution = st.slider(
            "Plot Resolution",
            min_value=20,
            max_value=100,
            value=50,
            help="Higher resolution = smoother plot but slower"
        )

    # Run optimization button
    st.markdown("---")

    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        run_button = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
    with run_col2:
        if st.button("🔄 New Random Start", use_container_width=True):
            st.session_state.random_seed = np.random.randint(0, 10000)
            st.rerun()

    if run_button:
        if not func_expr:
            st.error("Please enter a function expression.")
            return

        try:
            # Set random seed for reproducibility
            np.random.seed(st.session_state.random_seed)

            # Parse function
            objective_func = SafeExpressionParser.create_function(func_expr, var_names)

            # Parse constraint if provided
            constraint_func = None
            if constraint_expr and constraint_expr.strip():
                constraint_func = SafeExpressionParser.create_function(constraint_expr, var_names)

            # Create optimizer
            engine = OptimizationEngine(
                objective_func=objective_func,
                constraint_func=constraint_func,
                num_vars=num_vars,
                bounds=bounds
            )

            # Generate start point if random
            if start_point is None:
                start_point = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])

            # Run optimization
            with st.spinner("Optimizing..."):
                if find_type == 'Maximum':
                    result = engine.find_maximum(
                        method=optimization_method,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        x0=start_point
                    )
                else:
                    result = engine.optimize(
                        method=optimization_method,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        x0=start_point
                    )

            # Display results
            st.markdown("---")
            st.header("📈 Results")

            # Optimal point metrics
            result_cols = st.columns(num_vars + 2)
            for i, var in enumerate(var_names):
                with result_cols[i]:
                    st.metric(
                        label=f"Optimal {var}",
                        value=f"{result['x_optimal'][i]:.6f}"
                    )
            with result_cols[num_vars]:
                st.metric(
                    label=f"f* ({find_type})",
                    value=f"{result['f_optimal']:.6f}"
                )
            with result_cols[num_vars + 1]:
                st.metric(
                    label="Iterations",
                    value=f"{result['iterations']}"
                )

            # Visualizations for 2 variables
            if num_vars == 2:
                st.subheader("🎨 Visualizations")

                # Main contour plot with trajectory
                st.markdown("#### Optimization Trajectory")
                fig_contour = Visualizer.create_contour_with_trajectory(
                    objective_func,
                    x_range=bounds[0],
                    y_range=bounds[1],
                    path=result['path'],
                    resolution=resolution,
                    var_names=var_names,
                    constraint_func=constraint_func,
                    show_arrows=show_arrows
                )
                st.plotly_chart(fig_contour, use_container_width=True)

                # Two columns for 3D and convergence
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # 3D Surface Plot
                    st.markdown("#### 3D Surface")
                    fig_3d = Visualizer.create_3d_surface(
                        objective_func,
                        x_range=bounds[0],
                        y_range=bounds[1],
                        resolution=resolution,
                        var_names=var_names
                    )
                    fig_3d = Visualizer.add_optimization_path_3d(
                        fig_3d, result['path'], objective_func
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                with viz_col2:
                    if show_convergence and result.get('f_values'):
                        # Convergence Plot
                        st.markdown("#### Convergence")
                        fig_conv = Visualizer.create_convergence_plot(
                            result['f_values'],
                            learning_rate
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)

                    # Path statistics
                    st.markdown("#### Path Statistics")
                    path_array = np.array(result['path'])
                    total_distance = sum(
                        np.linalg.norm(path_array[i+1] - path_array[i])
                        for i in range(len(path_array)-1)
                    )
                    st.write(f"**Total path length:** {total_distance:.4f}")
                    st.write(f"**Start:** ({result['start_point'][0]:.4f}, {result['start_point'][1]:.4f})")
                    st.write(f"**End:** ({result['x_optimal'][0]:.4f}, {result['x_optimal'][1]:.4f})")

            elif num_vars == 3:
                st.subheader("🎨 Visualization (2D Slices)")
                st.info("For 3 variables, showing 2D slice at optimal c value.")

                # Create 2D slice at optimal c
                c_opt = result['x_optimal'][2]

                def slice_func(a, b):
                    return objective_func(a, b, c_opt)

                fig_3d = Visualizer.create_3d_surface(
                    slice_func,
                    x_range=bounds[0],
                    y_range=bounds[1],
                    resolution=resolution,
                    var_names=['a', 'b']
                )

                fig_3d.update_layout(
                    title=f'3D Surface of f(a, b, c={c_opt:.4f})'
                )

                st.plotly_chart(fig_3d, use_container_width=True)

            # Math Insights
            st.markdown("---")
            st.header("💡 Mathematical Insights")

            insights = MathInsightAnalyzer.analyze(
                result['x_optimal'],
                var_names,
                result['f_optimal'],
                constraint_expr
            )

            if insights:
                for insight in insights:
                    with st.expander(f"{insight['icon']} {insight['title']}", expanded=True):
                        st.markdown(f"**Description:** {insight['description']}")
                        st.markdown(f"**Suggestion:** {insight['suggestion']}")
                        st.info(f"🔧 Suggested Technique: **{insight['technique']}**")
            else:
                st.info("No special patterns detected. The optimal point may require numerical methods to prove.")

            # Optimization path details
            with st.expander("📍 Optimization Path Details"):
                st.write(f"**Method:** {optimization_method}")
                st.write(f"**Learning Rate:** {learning_rate}")
                st.write(f"**Number of iterations:** {result['iterations']}")
                st.write(f"**Starting point:** {result['start_point']}")
                st.write(f"**Final point:** {result['x_optimal']}")

                if result.get('f_values') and len(result['f_values']) > 1:
                    st.write(f"**Initial f value:** {result['f_values'][0]:.6f}")
                    st.write(f"**Final f value:** {result['f_values'][-1]:.6f}")
                    st.write(f"**Improvement:** {abs(result['f_values'][0] - result['f_values'][-1]):.6f}")

            # LaTeX representation
            st.markdown("---")
            st.subheader("📜 Summary")

            vars_str = ', '.join(var_names)
            opt_str = ', '.join([f"{var} = {val:.4f}" for var, val in zip(var_names, result['x_optimal'])])

            summary = f"""
            **Problem:** Find the {find_type.lower()} of $f({vars_str}) = {func_expr}$
            """
            if constraint_expr:
                summary += f"\n\n**Subject to:** ${constraint_expr} = 0$"

            summary += f"""

            **Solution:** The {find_type.lower()} value is **{result['f_optimal']:.6f}**

            **Equality Case (Điểm rơi):** ${opt_str}$
            """

            st.markdown(summary)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your expression syntax. Use ** for powers, and ensure all parentheses are balanced.")

    # Help section
    with st.expander("❓ Help & Examples"):
        st.markdown("""
        ### Syntax Guide

        | Operation | Syntax | Example |
        |-----------|--------|---------|
        | Addition | `+` | `a + b` |
        | Subtraction | `-` | `a - b` |
        | Multiplication | `*` | `a * b` |
        | Division | `/` | `a / b` |
        | Power | `**` or `^` | `a**2` or `a^2` |
        | Square Root | `sqrt()` | `sqrt(a)` |
        | Exponential | `exp()` | `exp(a)` |
        | Natural Log | `log()` | `log(a)` |
        | Trig Functions | `sin()`, `cos()`, `tan()` | `sin(a)` |
        | Constants | `pi`, `e` | `pi * a` |

        ### Learning Rate Effects

        | Learning Rate | Effect |
        |--------------|--------|
        | Too Low (0.001-0.01) | Slow convergence, smooth path |
        | Optimal (0.05-0.2) | Good balance of speed and stability |
        | Too High (0.5-1.0) | Fast but may zigzag or diverge |

        ### Classic Inequality Examples

        **1. AM-GM for 2 variables:**
        - Function: `a + b`
        - Constraint: `a * b - 1` (means ab = 1)
        - Minimum at a = b = 1, value = 2

        **2. Cauchy-Schwarz:**
        - Function: `(a**2 + b**2) * (1 + 1)`
        - Constraint: `a + b - 2`
        - Related to (a + b)² ≤ 2(a² + b²)

        **3. Nesbitt's Inequality (3 vars):**
        - Function: `a/(b+c) + b/(a+c) + c/(a+b)`
        - No constraint (use positive bounds)
        - Minimum at a = b = c, value = 3/2

        ### Tips

        - Use **Gradient Descent** method to see detailed optimization path
        - Click **"Randomize Start Point"** to see different paths converging to the same minimum
        - Adjust **Learning Rate** to see how it affects the path (zigzag vs smooth)
        - The **arrows** on the contour plot show the direction of descent
        - Điểm rơi (Equality Case) is the point where equality holds
        """)


if __name__ == "__main__":
    main()
