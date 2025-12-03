"""
Inequality Explorer - Interactive Min/Max Optimization Tool
===========================================================
"""

import streamlit as st
import numpy as np
import time
import warnings

import plotly.graph_objects as go

from latex_parser import LaTeXParser
from expression_parser import SafeExpressionParser
from optimization_engine import OptimizationEngine
from visualizer import Visualizer
from math_insights import MathInsightAnalyzer

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Inequality Explorer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for compact layout and smaller keyboard
st.markdown("""
<style>
/* Reduce overall padding */
.block-container { padding-top: 1rem; padding-bottom: 0; }

/* Scale down the MathLive iframe - small keyboard with visible input */
iframe[title="st_mathlive.math_box"] {
    transform: scale(0.55);
    transform-origin: top left;
    height: 520px !important;
    width: 182% !important;
    margin-bottom: -120px !important;
}

/* Reduce spacing between elements */
.stMarkdown { margin-bottom: -10px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Example expressions
EXAMPLES = {
    '1 var': {
        'Quadratic': ('a^{2}', '', 'a**2', ''),
        'Polynomial': ('a^{3}-3a^{2}+2a', '', 'a**3 - 3*a**2 + 2*a', ''),
        'AM-GM (a + 1/a)': ('a+\\frac{1}{a}', '', 'a + 1/a', ''),
        'Trigonometric': ('\\sin(a)+a', '', 'sin(a) + a', ''),
        'Custom': ('', '', '', ''),
    },
    '2 vars': {
        'Quadratic (no constraint)': ('a^{2}+b^{2}', '', 'a**2 + b**2', ''),
        'Quadratic + constraint': ('a^{2}+b^{2}', 'a+b-2', 'a**2 + b**2', 'a + b - 2'),
        'AM-GM Example': ('a+b+\\frac{1}{a}+\\frac{1}{b}', '', 'a + b + 1/a + 1/b', ''),
        'Rosenbrock': ('(1-a)^{2}+100(b-a^{2})^{2}', '', '(1-a)**2 + 100*(b-a**2)**2', ''),
        'Custom': ('', '', '', ''),
    },
    '3 vars': {
        'Quadratic 3D': ('a^{2}+b^{2}+c^{2}', '', 'a**2 + b**2 + c**2', ''),
        'Symmetric + constraint': ('a^{2}+b^{2}+c^{2}', 'a+b+c-3', 'a**2 + b**2 + c**2', 'a + b + c - 3'),
        'Custom': ('', '', '', ''),
    }
}


def detect_variables(expr: str) -> list:
    """
    Detect variables (a, b, c, d, x, y, z) from an expression.
    Returns sorted list of detected variable names.
    """
    import re
    if not expr:
        return ['a']

    # Clean the expression first (convert LaTeX to Python-like)
    cleaned = LaTeXParser.clean_latex_to_sympy(expr)

    # Known function names and constants to exclude
    exclude = {'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs', 'pi', 'e', 'E',
               'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'min', 'max', 'pow'}

    # Find all single letter variables (a-d, x-z are common math variables)
    # Pattern: standalone letters that are not part of function names
    found_vars = set()

    # Split by non-alphanumeric to find tokens
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cleaned)

    for token in tokens:
        # Check if it's a single letter variable (a-d or x-z)
        if len(token) == 1 and token.lower() in 'abcdxyz':
            found_vars.add(token.lower())
        # Also check for common variable patterns like x1, y2, etc.
        elif token.lower() not in exclude:
            # Check if first char is a variable letter
            if token[0].lower() in 'abcdxyz' and len(token) <= 2:
                found_vars.add(token.lower())

    # Sort variables: a, b, c, d first, then x, y, z
    priority = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'x': 4, 'y': 5, 'z': 6}
    sorted_vars = sorted(found_vars, key=lambda v: priority.get(v[0], 10))

    # Return at least 'a' if nothing found
    return sorted_vars if sorted_vars else ['a']


def main():
    # Session state
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = np.random.randint(0, 10000)

    st.markdown("## 📐 Inequality Explorer")

    # ===== SIDEBAR =====
    st.sidebar.header("Configuration")

    # Variable detection mode
    var_mode = st.sidebar.radio("Variables", ["Auto", "1", "2", "3"], horizontal=True, index=0)

    # Initialize var_names - will be updated after expression input
    if 'detected_vars' not in st.session_state:
        st.session_state.detected_vars = ['a', 'b']

    if var_mode == "Auto":
        var_names = st.session_state.detected_vars
        num_vars = len(var_names)
    else:
        num_vars = int(var_mode)
        var_names = ['a', 'b', 'c'][:num_vars]

    st.sidebar.markdown("---")
    optimization_method = st.sidebar.selectbox(
        "Method",
        ['gradient_descent', 'scipy'],
        format_func=lambda x: 'Gradient Descent' if x == 'gradient_descent' else 'SciPy (SLSQP)'
    )

    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001)
    iterations = st.sidebar.slider("Max Iterations", 10, 500, 50, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Bounds")
    bounds = []
    for var in var_names:
        cols = st.sidebar.columns(2)
        lb = cols[0].number_input(f"{var} min", value=-2.0, key=f"{var}_min")
        ub = cols[1].number_input(f"{var} max", value=5.0, key=f"{var}_max")
        bounds.append((lb, ub))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization")
    show_arrows = st.sidebar.checkbox("Show Arrows", True)
    show_convergence = st.sidebar.checkbox("Show Convergence", True)
    resolution = st.sidebar.slider("Resolution", 20, 100, 50)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation")
    animation_speed = st.sidebar.slider("Animation Speed (sec)", 0.5, 2.0, 0.5, 0.1)

    # Example selection - show all examples regardless of auto mode
    st.sidebar.markdown("---")
    all_examples = {}
    for cat, examples in EXAMPLES.items():
        for name, vals in examples.items():
            if name != 'Custom':
                all_examples[f"{cat}: {name}"] = vals
    all_examples["Custom"] = ('', '', '', '')

    selected_example = st.sidebar.selectbox("Load Example", list(all_examples.keys()))
    default_latex, default_constraint_latex, default_python, default_constraint_python = all_examples[selected_example]

    input_mode = st.sidebar.radio("Input Mode", ['Visual (LaTeX)', 'Text (Python)'], horizontal=True)
    find_type = st.sidebar.radio("Find", ['Minimum', 'Maximum'], horizontal=True)
    constraint_type = st.sidebar.selectbox("Constraint Type", ["= 0", ">= 0", "<= 0"], key="constraint_type_sidebar")

    # Show detected variables when in Auto mode
    if var_mode == "Auto":
        st.sidebar.info(f"Detected: {', '.join(var_names)} ({num_vars} var{'s' if num_vars > 1 else ''})")

    # Starting point
    start_mode = st.sidebar.radio("Start Point", ['Random', 'Custom'], horizontal=True)
    if start_mode == 'Custom':
        start_cols = st.sidebar.columns(num_vars)
        start_point = np.array([start_cols[i].number_input(f"{var}₀", value=2.0, key=f"start_{var}")
                               for i, var in enumerate(var_names)])
    else:
        start_point = None
        if st.sidebar.button("Randomize"):
            st.session_state.random_seed = np.random.randint(0, 10000)
            st.rerun()

    # ===== MAIN CONTENT =====
    if input_mode == 'Visual (LaTeX)':
        from st_mathlive import mathfield

        # Function and Constraint on same row (equal width)
        main_cols = st.columns([1, 1])

        with main_cols[0]:
            # Function input
            func_initial = default_latex if default_latex else "a^{2}+b^{2}"
            st.markdown(f"**f({', '.join(var_names)}) =**")
            func_result = mathfield(value=func_initial, title="", upright=False, mathml_preview=False, key="func_math")

            func_latex = ""
            if func_result:
                if isinstance(func_result, tuple):
                    func_latex = func_result[0] if func_result[0] else ""
                elif isinstance(func_result, list) and len(func_result) >= 1:
                    func_latex = func_result[0] if func_result[0] else ""
                else:
                    func_latex = str(func_result) if func_result else ""
            if not func_latex:
                func_latex = func_initial

            func_expr = LaTeXParser.clean_latex_to_sympy(func_latex) if func_latex else ""

        with main_cols[1]:
            # Constraint input with MathLive keyboard and inequality type
            st.markdown("**Constraint g (optional)**")
            constraint_initial = default_constraint_latex or ""
            constraint_result = mathfield(value=constraint_initial, title="", upright=False, mathml_preview=False, key="constraint_math")

            constraint_latex = ""
            if constraint_result:
                if isinstance(constraint_result, tuple):
                    constraint_latex = constraint_result[0] if constraint_result[0] else ""
                elif isinstance(constraint_result, list) and len(constraint_result) >= 1:
                    constraint_latex = constraint_result[0] if constraint_result[0] else ""
                else:
                    constraint_latex = str(constraint_result) if constraint_result else ""

            constraint_expr = LaTeXParser.clean_latex_to_sympy(constraint_latex) if constraint_latex else ""

        # Auto-detect variables from LaTeX expression
        if var_mode == "Auto" and func_latex:
            combined_expr = func_latex + " " + (constraint_latex or "")
            new_vars = detect_variables(combined_expr)
            if new_vars != st.session_state.detected_vars:
                st.session_state.detected_vars = new_vars
                st.rerun()

    else:
        # Text mode - function and constraint on same row
        main_cols = st.columns([3, 2])

        with main_cols[0]:
            func_expr = st.text_input(f"f({', '.join(var_names)}) =", value=default_python,
                                      placeholder="e.g., a**2 + b**2")

        with main_cols[1]:
            constraint_expr = st.text_input("Constraint g (optional)", value=default_constraint_python,
                                            placeholder="e.g., a + b - 2")

        # Auto-detect variables from Python expression
        if var_mode == "Auto" and func_expr:
            combined_expr = func_expr + " " + (constraint_expr or "")
            new_vars = detect_variables(combined_expr)
            if new_vars != st.session_state.detected_vars:
                st.session_state.detected_vars = new_vars
                st.rerun()

    # Run button
    col1, col2 = st.columns([4, 1])
    run_button = col1.button("Run Optimization", type="primary", use_container_width=True)
    if col2.button("New Start"):
        st.session_state.random_seed = np.random.randint(0, 10000)
        st.rerun()

    if run_button and func_expr:
        try:
            np.random.seed(st.session_state.random_seed)

            objective_func = SafeExpressionParser.create_function(func_expr, var_names)
            constraint_func = None
            # Map constraint type to engine format
            engine_constraint_type = 'eq'
            if constraint_type == ">= 0":
                engine_constraint_type = 'ineq_ge'
            elif constraint_type == "<= 0":
                engine_constraint_type = 'ineq_le'

            if constraint_expr and constraint_expr.strip():
                constraint_func = SafeExpressionParser.create_function(constraint_expr, var_names)

            engine = OptimizationEngine(objective_func, constraint_func, num_vars, bounds, engine_constraint_type)

            if start_point is None:
                # Use the engine's safe starting point generator
                start_point = engine._generate_safe_starting_point()

                # Additional search if we have a constraint
                if constraint_func:
                    # Check if current start point is valid (finite objective)
                    try:
                        current_f = objective_func(*start_point)
                        current_violation = abs(constraint_func(*start_point))
                        if np.isnan(current_f) or np.isinf(current_f) or current_f > 1e9:
                            current_f = float('inf')
                    except:
                        current_f = float('inf')
                        current_violation = float('inf')

                    best_start = start_point.copy()
                    best_f = current_f
                    best_violation = current_violation

                    # Helper function to evaluate a point
                    def is_better_point(test, best_f, best_violation):
                        try:
                            f_val = objective_func(*test)
                            violation = abs(constraint_func(*test))
                            # Only consider points where objective is finite
                            if np.isnan(f_val) or np.isinf(f_val) or f_val > 1e9:
                                return False, float('inf'), float('inf')
                            # Prefer lower violation first, then lower f value
                            if violation < best_violation - 0.01:
                                return True, f_val, violation
                            elif violation < best_violation + 0.01 and f_val < best_f:
                                return True, f_val, violation
                            return False, f_val, violation
                        except:
                            return False, float('inf'), float('inf')

                    # 1. Grid search for 2D problems (most reliable)
                    # Use angles far from 45 degrees for longer convergence path visualization
                    if num_vars == 2:
                        for angle in [np.radians(d) for d in [10, 15, 20, 70, 75, 80, 100, 110, 160, 170, 190, 200, 250, 260, 280, 290, 340, 350]]:
                            for r in [0.5, 1.0, 1.5, 2.0]:  # Don't include sqrt(2) values
                                test = np.array([r * np.cos(angle), r * np.sin(angle)])
                                is_better, f_val, violation = is_better_point(test, best_f, best_violation)
                                if is_better:
                                    best_f, best_violation = f_val, violation
                                    best_start = test.copy()

                    # 2. Random search avoiding near-zero values
                    for _ in range(200):
                        test = np.array([np.random.uniform(0.3, 3.0) * (1 if np.random.random() > 0.5 else -1)
                                        for _ in range(num_vars)])
                        is_better, f_val, violation = is_better_point(test, best_f, best_violation)
                        if is_better:
                            best_f, best_violation = f_val, violation
                            best_start = test.copy()

                    start_point = best_start

            with st.spinner("Optimizing..."):
                if find_type == 'Maximum':
                    result = engine.find_maximum(method=optimization_method, learning_rate=learning_rate,
                                                iterations=iterations, x0=start_point)
                else:
                    result = engine.optimize(method=optimization_method, learning_rate=learning_rate,
                                            iterations=iterations, x0=start_point)

            # Results
            st.markdown("---")
            st.subheader("Results")

            cols = st.columns(num_vars + 2)
            for i, var in enumerate(var_names):
                cols[i].metric(f"Optimal {var}", f"{result['x_optimal'][i]:.6f}")
            cols[num_vars].metric(f"f* ({find_type})", f"{result['f_optimal']:.6f}")
            cols[num_vars + 1].metric("Iterations", result['iterations'])

            # Visualizations
            if num_vars == 1:
                # 1 variable - show 2D plot with optimization path and animated version
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    fig_2d = Visualizer.create_2d_plot(objective_func, bounds[0], resolution, var_names[0])
                    fig_2d = Visualizer.add_optimization_path_2d(fig_2d, result['path'], objective_func, var_names[0])
                    st.plotly_chart(fig_2d, use_container_width=True)

                with viz_col2:
                    # Animated 1D plot with Play button
                    fig_anim = Visualizer.create_1d_contour_animation(
                        objective_func, bounds[0], result['path'], resolution,
                        var_names[0], int(animation_speed * 1000)
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)

                if show_convergence and result.get('f_values'):
                    fig_conv = Visualizer.create_convergence_plot(result['f_values'], learning_rate)
                    st.plotly_chart(fig_conv, use_container_width=True)

            elif num_vars == 2:
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    fig_3d = Visualizer.create_3d_surface(objective_func, bounds[0], bounds[1], resolution, var_names)
                    fig_3d = Visualizer.add_optimization_path_3d(fig_3d, result['path'], objective_func)
                    fig_3d.update_layout(height=450)
                    st.plotly_chart(fig_3d, use_container_width=True)

                with viz_col2:
                    # Create animated contour plot with Play button
                    fig_anim = Visualizer.create_autoplay_animation(
                        objective_func, bounds[0], bounds[1], result['path'], resolution,
                        var_names, constraint_func, int(animation_speed * 1000)
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)

                if show_convergence and result.get('f_values'):
                    fig_conv = Visualizer.create_convergence_plot(result['f_values'], learning_rate)
                    st.plotly_chart(fig_conv, use_container_width=True)

            else:
                # 3 variables - show convergence plot and variable trajectory plot
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    if show_convergence and result.get('f_values'):
                        fig_conv = Visualizer.create_convergence_plot(result['f_values'], learning_rate)
                        st.plotly_chart(fig_conv, use_container_width=True)

                with viz_col2:
                    # Plot each variable over iterations
                    if result.get('path') and len(result['path']) > 1:
                        path_array = np.array(result['path'])
                        fig_vars = go.Figure()
                        colors = ['blue', 'red', 'green']
                        for j, var in enumerate(var_names):
                            fig_vars.add_trace(go.Scatter(
                                x=list(range(len(path_array))),
                                y=path_array[:, j],
                                mode='lines+markers',
                                name=var,
                                line=dict(color=colors[j], width=2),
                                marker=dict(size=4)
                            ))
                        fig_vars.update_layout(
                            title='Variables over Iterations',
                            xaxis_title='Iteration',
                            yaxis_title='Value',
                            height=300,
                            showlegend=True
                        )
                        st.plotly_chart(fig_vars, use_container_width=True)

                # Show optimization path as a table
                if result.get('path') and len(result['path']) > 0:
                    st.markdown("**Optimization Path (first & last 5 points):**")
                    import pandas as pd
                    path_data = []
                    path = result['path']
                    indices = list(range(min(5, len(path)))) + list(range(max(0, len(path)-5), len(path)))
                    indices = sorted(set(indices))
                    for i in indices:
                        row = {'Step': i}
                        for j, var in enumerate(var_names):
                            row[var] = f"{path[i][j]:.6f}"
                        row['f(x)'] = f"{result['f_values'][i]:.6f}" if i < len(result['f_values']) else "N/A"
                        path_data.append(row)
                    st.dataframe(pd.DataFrame(path_data), use_container_width=True)

            # Insights
            st.markdown("---")
            st.subheader("Math Insights")
            insights = MathInsightAnalyzer.analyze(result['x_optimal'], var_names, result['f_optimal'], constraint_expr)

            if insights:
                for insight in insights:
                    with st.expander(f"{insight['title']}", expanded=True):
                        st.markdown(f"**{insight['description']}**")
                        st.markdown(f"Suggestion: {insight['suggestion']}")
                        st.info(f"Technique: {insight['technique']}")
            else:
                st.info("No special patterns detected.")

            # Summary
            st.markdown("---")
            vars_str = ', '.join(var_names)
            opt_str = ', '.join([f"{var} = {val:.4f}" for var, val in zip(var_names, result['x_optimal'])])
            constraint_str = ""
            if constraint_expr:
                constraint_str = f"**Subject to:** {constraint_expr} {constraint_type}"
            st.markdown(f"""
            **Problem:** Find the {find_type.lower()} of f({vars_str}) = {func_expr}
            {constraint_str}

            **Solution:** f* = **{result['f_optimal']:.6f}** at {opt_str}
            """)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Help
    with st.expander("Help"):
        st.markdown("""
        **Syntax:** Use `**` or `^` for powers, `sqrt()`, `sin()`, `cos()`, `exp()`, `log()`

        **Examples:**
        - `a**2 + b**2` - Simple quadratic
        - `a + b + 1/a + 1/b` - AM-GM type
        - Constraint `a + b - 2` means a + b = 2
        """)


if __name__ == "__main__":
    main()
