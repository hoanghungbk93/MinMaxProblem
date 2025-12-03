"""
Safe Expression Parser
=====================
Safely parse and evaluate mathematical expressions.
"""

import re
import numpy as np
from typing import Callable, List, Tuple

from latex_parser import LaTeXParser


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

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}"

        return True, "Valid"

    @classmethod
    def prepare_expression(cls, expr: str) -> str:
        """Prepare expression for evaluation (handle ^ for power)."""
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
                # Wrap to ensure float output
                def wrapped_func(*args):
                    result = numpy_func(*args)
                    # Convert SymPy/numpy objects to Python float
                    if hasattr(result, 'evalf'):
                        return float(result.evalf())
                    return float(result)
                return wrapped_func
        except Exception:
            pass

        # Fallback to eval-based parsing
        def func(*args):
            if len(args) != len(variables):
                raise ValueError(f"Expected {len(variables)} arguments, got {len(args)}")

            namespace = SAFE_MATH_FUNCTIONS.copy()
            for var, val in zip(variables, args):
                namespace[var] = val

            try:
                result = eval(prepared_expr, {"__builtins__": {}}, namespace)
                # Convert SymPy/numpy objects to Python float
                if hasattr(result, 'evalf'):
                    return float(result.evalf())
                return float(result)
            except Exception as e:
                raise ValueError(f"Error evaluating expression: {e}")

        return func
