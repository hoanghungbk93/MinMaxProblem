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
        - a^{2} + b^{2} = 1 -> a^{2} + b^{2} - 1 (for constraints)
        """
        if not latex_str or not latex_str.strip():
            return ""

        result = latex_str.strip()

        # Handle equations: convert "expr = value" to "expr - value"
        # This allows users to enter constraints like "a^2 + b^2 = 1"
        if '=' in result and not any(op in result for op in ['>=', '<=', '!=']):
            parts = result.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                if right and right != '0':
                    result = f"({left}) - ({right})"
                else:
                    result = left

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

        # Handle logarithm with base: \log_{10}x -> log10(x), \log_{2}x -> log2(x)
        # Must be handled BEFORE subscript removal
        # Must be handled before general \log replacement
        # Pattern: \log_{base}(arg) or \log_{base} arg or \log_{base}arg
        def replace_log_base(match):
            base = match.group(1)
            arg = match.group(2) if match.group(2) else ''
            if base == '10':
                return f'log10({arg})' if arg else 'log10'
            elif base == '2':
                return f'log2({arg})' if arg else 'log2'
            else:
                # General base: log_b(x) = log(x)/log(b)
                if arg:
                    return f'(log({arg})/log({base}))'
                else:
                    return f'log'  # fallback

        # \log_{base}{arg} - with braces around argument
        result = re.sub(r'\\log_\{([^{}]+)\}\{([^{}]+)\}', replace_log_base, result)
        # \log_{base}(arg) - with parentheses
        result = re.sub(r'\\log_\{([^{}]+)\}\(([^()]+)\)', lambda m: replace_log_base(m), result)
        # \log_{base}x - single letter/digit argument (no braces)
        result = re.sub(r'\\log_\{([^{}]+)\}([a-zA-Z0-9])', replace_log_base, result)
        # \log_{base} followed by space then variable - like \log_{10} a
        result = re.sub(r'\\log_\{([^{}]+)\}\s+([a-zA-Z0-9])', replace_log_base, result)
        # \log_{base} followed by space or end - standalone, will need implicit multiply later
        result = re.sub(r'\\log_\{([^{}]+)\}(?=\s|$|[+\-*/)])', lambda m: f'log{m.group(1)}' if m.group(1) in ['10', '2'] else 'log', result)
        # \log_base (no braces around base) - simpler form like \log_10 a
        result = re.sub(r'\\log_(\d+)\s*([a-zA-Z])', lambda m: f'log{m.group(1)}({m.group(2)})' if m.group(1) in ['10', '2'] else f'(log({m.group(2)})/log({m.group(1)}))', result)

        # Handle subscripts (remove for now or convert to variable naming)
        # This must come AFTER log base handling
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

        # Add implicit multiplication: 2a -> 2*a, a b -> a*b, ab -> a*b
        # But NOT for function names like sqrt, sin, cos, etc.
        func_names = ['sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'log2', 'ln', 'exp', 'abs', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'pi']
        # Variables are single lowercase letters
        variables = set('abcdxyz')

        # Number followed by letter
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
        # Closing paren followed by letter or number
        result = re.sub(r'\)([a-zA-Z0-9])', r')*\1', result)

        # Single letter variable followed by another single letter variable: ab -> a*b
        # This handles implicit multiplication between variables
        def add_mult_between_vars(match):
            full = match.group(0)
            # Don't split function names
            for func in func_names:
                if full == func:
                    return full
            # Split consecutive single-letter variables with *
            result_chars = []
            for i, c in enumerate(full):
                result_chars.append(c)
                if i < len(full) - 1:
                    # Both current and next are single-letter variables
                    if c.lower() in variables and full[i+1].lower() in variables:
                        result_chars.append('*')
            return ''.join(result_chars)

        # Match sequences of 2+ lowercase letters that might be implicit multiplication
        result = re.sub(r'[a-z]{2,}', add_mult_between_vars, result)

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

        # Convert uppercase variable letters to lowercase (A->a, B->b, C->c)
        # But preserve function names and constants
        def lowercase_vars(match):
            char = match.group(0)
            # Don't convert E (Euler's number) or other constants
            if char == 'E':
                return char
            return char.lower()

        # Only convert standalone uppercase letters that are likely variables
        result = re.sub(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', lowercase_vars, result)

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
