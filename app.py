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
import time
from sympy import sympify, lambdify, symbols, sqrt, sin, cos, tan, exp, log, pi, E
from sympy.parsing.latex import parse_latex

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
# LATEX TO SYMPY PARSER
# =============================================================================

class LaTeXParser:
    """Convert LaTeX expressions from MathLive to SymPy-compatible format."""

    @staticmethod
    def clean_latex_to_sympy(latex_str: str) -> str:
        """
        Convert LaTeX string to a SymPy-compatible Python expression.

        Examples:
        - \frac{a^{2}}{b} -> ((a)**(2))/(b)
        - \sqrt{a} -> sqrt(a)
        - a^{2} -> (a)**(2)
        - \cdot -> *
        - \sin(x) -> sin(x)
        """
        if not latex_str or not latex_str.strip():
            return ""

        result = latex_str.strip()

        # Remove display math delimiters if present
        result = re.sub(r'^\$\$?|\$\$?$', '', result)
        result = re.sub(r'^\\\[|\\]$', '', result)

        # Handle fractions: \frac{num}{denom} -> (num)/(denom)
        # This regex handles nested braces properly
        def replace_frac(match):
            # Find matching braces for numerator
            full = match.group(0)
            rest = full[5:]  # Remove \frac

            # Extract numerator (content within first {})
            num, rest = LaTeXParser._extract_braces(rest)
            # Extract denominator (content within second {})
            denom, rest = LaTeXParser._extract_braces(rest)

            return f"(({num}))/({denom}){rest}"

        # Replace fractions iteratively (handle nested fractions)
        while r'\frac' in result:
            new_result = re.sub(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                               lambda m: f"(({m.group(1)}))/({m.group(2)})", result)
            if new_result == result:
                break
            result = new_result

        # Handle square root: \sqrt{x} -> sqrt(x)
        result = re.sub(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'sqrt(\1)', result)

        # Handle nth root: \sqrt[n]{x} -> (x)**(1/(n))
        result = re.sub(r'\\sqrt\[([^\]]+)\]\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                       r'((\2))**(1/(\1))', result)

        # Handle power with braces: a^{2} -> (a)**(2)
        result = re.sub(r'\^{([^{}]+)}', r'**(\1)', result)

        # Handle simple power: a^2 -> (a)**(2) - but only single char/digit
        result = re.sub(r'\^(\d+)', r'**(\1)', result)
        result = re.sub(r'\^([a-zA-Z])', r'**(\1)', result)

        # Handle subscripts (remove for now or convert to variable naming)
        result = re.sub(r'_\{([^{}]+)\}', r'_\1', result)
        result = re.sub(r'_(\d+)', r'_\1', result)

        # Handle trigonometric and other functions
        result = re.sub(r'\\sin\s*', 'sin', result)
        result = re.sub(r'\\cos\s*', 'cos', result)
        result = re.sub(r'\\tan\s*', 'tan', result)
        result = re.sub(r'\\log\s*', 'log', result)
        result = re.sub(r'\\ln\s*', 'log', result)
        result = re.sub(r'\\exp\s*', 'exp', result)
        result = re.sub(r'\\abs\s*', 'abs', result)

        # Handle |x| absolute value notation
        result = re.sub(r'\|([^|]+)\|', r'abs(\1)', result)
        result = re.sub(r'\\left\|([^|]+)\\right\|', r'abs(\1)', result)

        # Handle multiplication operators
        result = result.replace(r'\cdot', '*')
        result = result.replace(r'\times', '*')
        result = result.replace(r'\ast', '*')

        # Handle division
        result = result.replace(r'\div', '/')

        # Handle constants
        result = result.replace(r'\pi', 'pi')
        result = re.sub(r'\\mathrm\{e\}', 'E', result)

        # Handle parentheses
        result = result.replace(r'\left(', '(')
        result = result.replace(r'\right)', ')')
        result = result.replace(r'\left[', '(')
        result = result.replace(r'\right]', ')')
        result = result.replace(r'\{', '(')
        result = result.replace(r'\}', ')')

        # Replace remaining curly braces with parentheses (avoid Python set literal)
        result = result.replace('{', '(')
        result = result.replace('}', ')')

        # Handle plus/minus
        result = result.replace(r'\pm', '+')
        result = result.replace(r'\mp', '-')

        # Handle comparison operators (for constraints)
        result = result.replace(r'\geq', '>=')
        result = result.replace(r'\leq', '<=')
        result = result.replace(r'\ge', '>=')
        result = result.replace(r'\le', '<=')
        result = result.replace(r'\neq', '!=')

        # Remove remaining backslashes and LaTeX commands
        result = re.sub(r'\\[a-zA-Z]+', '', result)

        # Add implicit multiplication: 2a -> 2*a, a b -> a*b
        # But NOT for function names like sqrt, sin, cos, etc.
        func_names = ['sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'abs', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']

        # Number followed by letter
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
        # Letter followed by number (for things like a2 meaning a*2, less common)
        # result = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', result)
        # Closing paren followed by letter or number
        result = re.sub(r'\)([a-zA-Z0-9])', r')*\1', result)

        # Letter/number followed by opening paren - but exclude function names
        def add_mult_before_paren(match):
            before = match.group(1)
            # Check if this ends with a function name
            for func in func_names:
                if before.endswith(func):
                    return match.group(0)  # Don't add multiplication
            return before + '*('

        result = re.sub(r'([a-zA-Z0-9]+)\(', add_mult_before_paren, result)

        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    @staticmethod
    def _extract_braces(s: str) -> Tuple[str, str]:
        """Extract content within first {} and return (content, remaining)."""
        if not s or s[0] != '{':
            return '', s

        depth = 0
        for i, c in enumerate(s):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[1:i], s[i+1:]

        return s[1:], ''

    @staticmethod
    def latex_to_sympy_expr(latex_str: str, var_names: List[str]):
        """
        Parse LaTeX string and return a SymPy expression.
        Falls back to clean_latex_to_sympy if parse_latex fails.
        """
        if not latex_str or not latex_str.strip():
            return None

        try:
            # Try SymPy's built-in LaTeX parser first
            expr = parse_latex(latex_str)
            return expr
        except Exception:
            pass

        # Fallback: clean the LaTeX and use sympify
        try:
            cleaned = LaTeXParser.clean_latex_to_sympy(latex_str)
            if cleaned:
                # Create symbols
                syms = symbols(' '.join(var_names))
                if len(var_names) == 1:
                    syms = [syms]
                local_dict = {name: sym for name, sym in zip(var_names, syms)}
                local_dict['pi'] = pi
                local_dict['E'] = E
                local_dict['e'] = E

                expr = sympify(cleaned, locals=local_dict)
                return expr
        except Exception:
            pass

        return None

    @staticmethod
    def create_numpy_function(latex_str: str, var_names: List[str]) -> Tuple[Callable, str, str]:
        """
        Convert LaTeX to a NumPy-compatible callable function.

        Returns: (function, cleaned_python_expr, error_message)
        """
        if not latex_str or not latex_str.strip():
            return None, "", "Expression is empty"

        # First clean the LaTeX
        cleaned = LaTeXParser.clean_latex_to_sympy(latex_str)

        try:
            # Create SymPy symbols
            syms = symbols(' '.join(var_names))
            if len(var_names) == 1:
                syms = (syms,)

            local_dict = {name: sym for name, sym in zip(var_names, syms)}
            local_dict['pi'] = pi
            local_dict['E'] = E
            local_dict['e'] = E

            # Parse the cleaned expression
            expr = sympify(cleaned, locals=local_dict)

            # Create numpy-compatible function
            numpy_func = lambdify(syms, expr, modules=['numpy'])

            return numpy_func, cleaned, ""

        except Exception as e:
            return None, cleaned, str(e)


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

        # First try using SymPy for more robust parsing
        try:
            numpy_func, _, error = LaTeXParser.create_numpy_function(prepared_expr, variables)
            if numpy_func is not None:
                return numpy_func
        except Exception:
            pass

        # Fallback to eval-based parsing
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

    def _objective_with_penalty(self, x: np.ndarray, penalty_weight: float = 100000) -> float:
        """Objective with constraint penalty (higher weight for better constraint satisfaction)."""
        obj_val = self._objective_wrapper(x)

        if self.constraint_func is not None:
            try:
                constraint_val = self.constraint_func(*x)
                # Use very high penalty to enforce constraint
                penalty = penalty_weight * constraint_val ** 2
                obj_val += penalty
            except:
                obj_val += 1e10

        return obj_val

    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                         iterations: int = 100, epsilon: float = 1e-6) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Gradient Descent: θ_{t+1} = θ_t - η * ∇f(θ_t)
        Reference: https://machinelearningcoban.com/2017/01/12/gradientdescent/
        """
        x = x0.copy().astype(float)
        path = [x.copy()]
        f_values = [self._objective_wrapper(x)]

        for iter_num in range(iterations):
            # Compute gradient numerically: ∇f ≈ [f(x+ε) - f(x-ε)] / (2ε)
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                x_minus = x.copy()
                x_minus[i] -= epsilon

                # Use penalty function if constraint exists
                if self.constraint_func is not None:
                    grad[i] = (self._objective_with_penalty(x_plus) -
                              self._objective_with_penalty(x_minus)) / (2 * epsilon)
                else:
                    # Pure gradient descent for unconstrained
                    grad[i] = (self._objective_wrapper(x_plus) -
                              self._objective_wrapper(x_minus)) / (2 * epsilon)

            # Gradient descent update: x = x - η * ∇f(x)
            x_new = x - learning_rate * grad

            # Project back to bounds
            for i, (lb, ub) in enumerate(self.bounds):
                x_new[i] = np.clip(x_new[i], lb, ub)

            x = x_new
            path.append(x.copy())
            f_values.append(self._objective_wrapper(x))

            # Check convergence: ||∇f|| < ε
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
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
    def create_animated_contour_frame(func: Callable, x_range: Tuple[float, float],
                                       y_range: Tuple[float, float],
                                       path: List[np.ndarray],
                                       current_step: int,
                                       resolution: int = 50,
                                       var_names: List[str] = ['a', 'b'],
                                       constraint_func: Callable = None) -> go.Figure:
        """Create a single frame of the animated contour plot."""
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
                contours=dict(start=0, end=0, size=0.01, coloring='lines'),
                line=dict(color='red', width=4, dash='dash'),
                showscale=False,
                name='Constraint (g=0)'
            ))

        # Get path up to current step
        if path is not None and len(path) >= 1:
            path_array = np.array(path[:current_step + 1])
            n_points = len(path_array)

            if n_points >= 2:
                # Draw path segments with gradient color
                for i in range(n_points - 1):
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

                # Path points
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

            # Start point
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

            # Current point (animated marker)
            fig.add_trace(go.Scatter(
                x=[path_array[-1, 0]],
                y=[path_array[-1, 1]],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='circle',
                           line=dict(color='darkred', width=3)),
                text=[f'Step {current_step}'],
                textposition='bottom center',
                textfont=dict(size=12, color='darkred', family='Arial Black'),
                name='Current Point'
            ))

        fig.update_layout(
            title=dict(
                text=f'Optimization Progress - Step {current_step}/{len(path)-1 if path else 0}',
                font=dict(size=16)
            ),
            xaxis_title=var_names[0],
            yaxis_title=var_names[1],
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
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

    st.markdown("## 📐 Inequality Explorer")

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
        value=50,  # Lower default for faster animation
        step=10,
        help="Maximum number of optimization iterations"
    )

    # Variable bounds
    st.sidebar.subheader("Variable Bounds")
    bounds = []
    for var in var_names:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lb = st.number_input(f"{var} min", value=-2.0, step=0.1, key=f"{var}_min")
        with col2:
            ub = st.number_input(f"{var} max", value=5.0, step=0.1, key=f"{var}_max")
        bounds.append((lb, ub))

    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_arrows = st.sidebar.checkbox("Show Direction Arrows", value=True,
                                      help="Display arrows showing descent direction on contour plot")
    show_convergence = st.sidebar.checkbox("Show Convergence Plot", value=True,
                                           help="Display how f(a,b) changes over iterations")

    # Animation options
    st.sidebar.subheader("Animation")
    animate_path = st.sidebar.checkbox("Animate Optimization", value=False,
                                       help="Watch the optimization path update step-by-step")
    animation_speed = st.sidebar.slider(
        "Animation Speed (seconds)",
        min_value=0.05,
        max_value=2.0,
        value=0.3,
        step=0.05,
        help="Delay between each step (lower = faster)",
        disabled=not animate_path
    )
    steps_per_frame = st.sidebar.slider(
        "Steps per Frame",
        min_value=1,
        max_value=20,
        value=1,  # Default to 1 to show EVERY step
        step=1,
        help="1 = show every step, higher = skip steps",
        disabled=not animate_path
    )

    # ===== SIDEBAR: All configuration =====
    # Example expressions (LaTeX format for MathLive, Python format as fallback)
    examples = {
        '2 vars': {
            'Quadratic (no constraint)': ('a^{2}+b^{2}', '', 'a**2 + b**2', ''),
            'Quadratic + constraint': ('a^{2}+b^{2}', 'a+b-2', 'a**2 + b**2', 'a + b - 2'),
            'AM-GM Example': ('a+b+\\frac{1}{a}+\\frac{1}{b}', '', 'a + b + 1/a + 1/b', ''),
            'Rosenbrock': ('(1-a)^{2}+100(b-a^{2})^{2}', '', '(1-a)**2 + 100*(b-a**2)**2', ''),
            'Himmelblau': ('(a^{2}+b-11)^{2}+(a+b^{2}-7)^{2}', '', '(a**2 + b - 11)**2 + (a + b**2 - 7)**2', ''),
            'Custom': ('', '', '', ''),
        },
        '3 vars': {
            'Quadratic 3D': ('a^{2}+b^{2}+c^{2}', '', 'a**2 + b**2 + c**2', ''),
            'Symmetric + constraint': ('a^{2}+b^{2}+c^{2}', 'a+b+c-3', 'a**2 + b**2 + c**2', 'a + b + c - 3'),
            'AM-GM 3 vars': ('a+b+c+\\frac{1}{a}+\\frac{1}{b}+\\frac{1}{c}', '', 'a + b + c + 1/a + 1/b + 1/c', ''),
            'Custom': ('', '', '', ''),
        }
    }

    st.sidebar.subheader("Load Example")
    example_key = '2 vars' if num_vars == 2 else '3 vars'
    selected_example = st.sidebar.selectbox(
        "Example",
        options=list(examples[example_key].keys()),
        index=0,
        label_visibility="collapsed"
    )
    default_latex, default_constraint_latex, default_python, default_constraint_python = examples[example_key][selected_example]

    st.sidebar.subheader("Input Mode")
    input_mode = st.sidebar.radio(
        "Input Mode",
        options=['Visual (LaTeX)', 'Text (Python)'],
        horizontal=True,
        help="Visual: Math editor. Text: Python expressions.",
        label_visibility="collapsed"
    )

    st.sidebar.subheader("Find")
    find_type = st.sidebar.radio(
        "Find",
        options=['Minimum', 'Maximum'],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.sidebar.subheader("Starting Point")
    start_mode = st.sidebar.radio(
        "Starting Point Mode",
        options=['Random', 'Custom'],
        horizontal=True,
        label_visibility="collapsed"
    )

    if start_mode == 'Custom':
        start_cols = st.sidebar.columns(num_vars)
        start_point = []
        for i, var in enumerate(var_names):
            with start_cols[i]:
                val = st.number_input(f"{var}₀", value=2.0, step=0.1, key=f"start_{var}")
                start_point.append(val)
        start_point = np.array(start_point)
    else:
        start_point = None
        if st.sidebar.button("🎲 Randomize"):
            st.session_state.random_seed = np.random.randint(0, 10000)
            st.rerun()
        st.sidebar.caption(f"Seed: {st.session_state.random_seed}")

    st.sidebar.subheader("Plot Resolution")
    resolution = st.sidebar.slider(
        "Resolution",
        min_value=20,
        max_value=100,
        value=50,
        label_visibility="collapsed"
    )

    # ===== MAIN CONTENT: Compact formula editor =====
    if input_mode == 'Visual (LaTeX)':
        from st_mathlive import mathfield

        # Compact header with function label inline
        func_initial = default_latex if default_latex else "a^{2}+b^{2}"

        col_label, col_editor = st.columns([1, 5])
        with col_label:
            st.markdown(f"**f({', '.join(var_names)}) =**")
        with col_editor:
            func_result = mathfield(value=func_initial)

        if func_result:
            if isinstance(func_result, tuple):
                func_latex, func_mathml = func_result
            elif isinstance(func_result, list) and len(func_result) >= 1:
                func_latex = func_result[0] if func_result[0] else ""
            else:
                func_latex = str(func_result) if func_result else ""
        else:
            func_latex = func_initial

        if func_latex and func_latex.strip():
            func_expr_cleaned = LaTeXParser.clean_latex_to_sympy(func_latex)
        else:
            func_expr_cleaned = ""

        # Constraint on same row style
        col_clabel, col_cinput = st.columns([1, 5])
        with col_clabel:
            st.markdown("**g =** *(optional)*")
        with col_cinput:
            constraint_initial = default_constraint_latex if default_constraint_latex else ""
            constraint_latex = st.text_input(
                "Constraint",
                value=constraint_initial,
                placeholder="e.g., a + b - 2 (constraint = 0)",
                label_visibility="collapsed"
            )

        if constraint_latex and constraint_latex.strip():
            constraint_expr_cleaned = LaTeXParser.clean_latex_to_sympy(constraint_latex)
        else:
            constraint_expr_cleaned = ""

        func_expr = func_expr_cleaned
        constraint_expr = constraint_expr_cleaned

    else:
        # Text mode - compact layout
        col_label, col_editor = st.columns([1, 5])
        with col_label:
            st.markdown(f"**f({', '.join(var_names)}) =**")
        with col_editor:
            func_expr = st.text_input(
                "Function",
                value=default_python,
                placeholder=f"e.g., a**2 + b**2 + 1/a + 1/b",
                label_visibility="collapsed"
            )

        col_clabel, col_cinput = st.columns([1, 5])
        with col_clabel:
            st.markdown("**g =** *(optional)*")
        with col_cinput:
            constraint_expr = st.text_input(
                "Constraint (= 0)",
                value=default_constraint_python,
                placeholder="e.g., a + b - 2",
                label_visibility="collapsed"
            )

    # Run optimization button (compact)
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

                # If there's a constraint, try to find a starting point that satisfies it
                if constraint_func is not None:
                    # For simple linear constraints like a + b = c, adjust start point
                    # Try to project onto constraint by simple search
                    best_start = start_point.copy()
                    best_violation = abs(constraint_func(*start_point))

                    for _ in range(100):
                        test_point = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
                        violation = abs(constraint_func(*test_point))
                        if violation < best_violation:
                            best_violation = violation
                            best_start = test_point

                    start_point = best_start

            # Run optimization
            # Use scipy for constrained optimization (much better), gradient descent for unconstrained
            effective_method = optimization_method
            if constraint_func is not None and optimization_method == 'gradient_descent':
                st.warning("Note: Using SciPy for constrained optimization (more accurate). Gradient descent with constraints uses penalty method which may be less precise.")

            with st.spinner("Optimizing..."):
                if find_type == 'Maximum':
                    result = engine.find_maximum(
                        method=effective_method,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        x0=start_point
                    )
                else:
                    result = engine.optimize(
                        method=effective_method,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        x0=start_point
                    )

            # Verify constraint satisfaction
            if constraint_func is not None:
                constraint_violation = abs(constraint_func(*result['x_optimal']))
                if constraint_violation > 0.01:
                    st.warning(f"Constraint violation: {constraint_violation:.4f}. Try using SciPy method or adjusting parameters.")

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

                # Two columns: 3D Surface (left) and Contour/Animation (right)
                viz_col1, viz_col2 = st.columns([1, 1])

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
                    fig_3d.update_layout(height=500)
                    st.plotly_chart(fig_3d, use_container_width=True)

                with viz_col2:
                    st.markdown("#### Optimization Trajectory")
                    if animate_path and len(result['path']) > 1:
                        # Animated visualization using matplotlib (faster updates)
                        import matplotlib.pyplot as plt
                        import matplotlib.patches as mpatches

                        total_steps = len(result['path'])
                        path_array = np.array(result['path'])

                        st.info(f"🎬 Animating {total_steps} steps...")

                        # Pre-compute contour data ONCE
                        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
                        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
                        X, Y = np.meshgrid(x, y)
                        Z = np.zeros_like(X)
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    Z[i, j] = objective_func(X[i, j], Y[i, j])
                                except:
                                    Z[i, j] = np.nan

                        # Create placeholders
                        plot_placeholder = st.empty()
                        status_text = st.empty()

                        # Pre-compute constraint if exists
                        C = None
                        if constraint_func is not None:
                            C = np.zeros_like(X)
                            for i in range(X.shape[0]):
                                for j in range(X.shape[1]):
                                    try:
                                        C[i, j] = constraint_func(X[i, j], Y[i, j])
                                    except:
                                        C[i, j] = np.nan

                        # Get start and end points
                        start_point = path_array[0]
                        end_point = path_array[-1]  # Minimum point (pre-calculated)

                        # Animation loop - ADD new point and line each frame
                        for step in range(1, total_steps + 1, steps_per_frame):
                            current_step = min(step, total_steps)

                            # Create SMALLER figure for side-by-side layout
                            fig, ax = plt.subplots(figsize=(6, 5))

                            # Draw contour background
                            ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
                            contour = ax.contour(X, Y, Z, levels=20, colors='blue', linewidths=0.5)
                            ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')

                            # Draw constraint if exists
                            if C is not None:
                                ax.contour(X, Y, C, levels=[0], colors='red', linewidths=2, linestyles='--')

                            # Always show START point (green)
                            ax.scatter(start_point[0], start_point[1],
                                      c='limegreen', s=150, marker='o', edgecolors='darkgreen',
                                      linewidths=2, zorder=15, label='Start')

                            # Always show MINIMUM point (gold star) - pre-calculated
                            ax.scatter(end_point[0], end_point[1],
                                      c='gold', s=200, marker='*', edgecolors='darkorange',
                                      linewidths=2, zorder=15, label='Min')

                            # Draw path UP TO current step (lines + points)
                            current_path = path_array[:current_step]

                            if len(current_path) >= 2:
                                # Draw LINE from start to current (red line)
                                ax.plot(current_path[:, 0], current_path[:, 1],
                                       'r-', linewidth=2, zorder=10)

                                # Draw all intermediate points (small red dots)
                                ax.scatter(current_path[1:-1, 0], current_path[1:-1, 1],
                                          c='red', s=30, zorder=12, alpha=0.8)

                            # Draw CURRENT point (larger, highlighted)
                            current_point = current_path[-1]
                            ax.scatter(current_point[0], current_point[1],
                                      c='red', s=150, marker='o', edgecolors='darkred',
                                      linewidths=2, zorder=20)

                            # Calculate gradient norm
                            try:
                                current_f = objective_func(*current_point)
                                eps = 1e-5
                                grad = np.zeros(2)
                                for i in range(2):
                                    p_plus = current_point.copy()
                                    p_plus[i] += eps
                                    p_minus = current_point.copy()
                                    p_minus[i] -= eps
                                    grad[i] = (objective_func(*p_plus) - objective_func(*p_minus)) / (2 * eps)
                                grad_norm = np.linalg.norm(grad)
                            except:
                                current_f = 0
                                grad_norm = 0

                            # Title
                            ax.set_xlabel(var_names[0], fontsize=10)
                            ax.set_ylabel(var_names[1], fontsize=10)
                            ax.set_title(f'η={learning_rate}; iter={current_step-1}/{total_steps-1}; ||∇f||={grad_norm:.3f}',
                                        fontsize=10)
                            ax.legend(loc='upper right', fontsize=8)

                            plt.tight_layout()

                            # Update plot
                            plot_placeholder.pyplot(fig)
                            plt.close(fig)

                            # Update status text
                            status_text.markdown(
                                f"**iter {current_step-1}/{total_steps-1}** | "
                                f"f = {current_f:.4f}"
                            )

                            time.sleep(animation_speed)

                        # Final status
                        status_text.markdown("**✅ Complete!**")

                        # Show final frame
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
                        contour = ax.contour(X, Y, Z, levels=20, colors='blue', linewidths=0.5)
                        ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')

                        if constraint_func is not None:
                            ax.contour(X, Y, C, levels=[0], colors='red', linewidths=2, linestyles='--')

                        # Full path
                        ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2)
                        ax.scatter(path_array[1:-1, 0], path_array[1:-1, 1], c='red', s=20, zorder=5, alpha=0.7)

                        # Start (green)
                        ax.scatter(path_array[0, 0], path_array[0, 1],
                                  c='limegreen', s=150, marker='o', edgecolors='darkgreen',
                                  linewidths=2, zorder=10, label='Start')

                        # End (gold star)
                        ax.scatter(path_array[-1, 0], path_array[-1, 1],
                                  c='gold', s=250, marker='*', edgecolors='darkorange',
                                  linewidths=2, zorder=10, label='Min ★')

                        ax.set_xlabel(var_names[0], fontsize=10)
                        ax.set_ylabel(var_names[1], fontsize=10)
                        ax.set_title(f'f* = {result["f_optimal"]:.4f}', fontsize=10)
                        ax.legend(loc='upper right', fontsize=8)
                        plt.tight_layout()

                        plot_placeholder.pyplot(fig)
                        plt.close(fig)

                    else:
                        # Static visualization (no animation)
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
                        fig_contour.update_layout(height=500)
                        st.plotly_chart(fig_contour, use_container_width=True)

                # Convergence plot below
                if show_convergence and result.get('f_values'):
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
