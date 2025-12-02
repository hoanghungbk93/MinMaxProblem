"""
LaTeX to SymPy Parser
====================
Convert LaTeX expressions from MathLive to SymPy-compatible format.
"""

import re
from typing import Callable, List, Tuple
from sympy import sympify, lambdify, symbols, sqrt, sin, cos, tan, exp, log, pi, E
from sympy.parsing.latex import parse_latex


class LaTeXParser:
    """Convert LaTeX expressions from MathLive to SymPy-compatible format."""

    @staticmethod
    def clean_latex_to_sympy(latex_str: str) -> str:
        """
        Convert LaTeX string to a SymPy-compatible Python expression.

        Examples:
        - \\frac{a^{2}}{b} -> ((a)**(2))/(b)
        - \\sqrt{a} -> sqrt(a)
        - a^{2} -> (a)**(2)
        - \\cdot -> *
        - \\sin(x) -> sin(x)
        """
        if not latex_str or not latex_str.strip():
            return ""

        result = latex_str.strip()

        # Remove display math delimiters if present
        result = re.sub(r'^\$\$?|\$\$?$', '', result)
        result = re.sub(r'^\\\[|\\]$', '', result)

        # Handle fractions: \frac{num}{denom} -> (num)/(denom)
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
        # Closing paren followed by letter or number
        result = re.sub(r'\)([a-zA-Z0-9])', r')*\1', result)

        # Letter/number followed by opening paren - but exclude function names
        def add_mult_before_paren(match):
            before = match.group(1)
            for func in func_names:
                if before.endswith(func):
                    return match.group(0)
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
            expr = parse_latex(latex_str)
            return expr
        except Exception:
            pass

        try:
            cleaned = LaTeXParser.clean_latex_to_sympy(latex_str)
            if cleaned:
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

        cleaned = LaTeXParser.clean_latex_to_sympy(latex_str)

        try:
            syms = symbols(' '.join(var_names))
            if len(var_names) == 1:
                syms = (syms,)

            local_dict = {name: sym for name, sym in zip(var_names, syms)}
            local_dict['pi'] = pi
            local_dict['E'] = E
            local_dict['e'] = E

            expr = sympify(cleaned, locals=local_dict)
            numpy_func = lambdify(syms, expr, modules=['numpy'])

            return numpy_func, cleaned, ""

        except Exception as e:
            return None, cleaned, str(e)
