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

/* Scale down the MathLive iframe - keep formula visible, shrink keyboard */
iframe[title="st_mathlive.math_box"] {
    transform: scale(0.85);
    transform-origin: top left;
    height: 380px !important;
    width: 118% !important;
}

/* Reduce spacing between elements */
.stMarkdown { margin-bottom: -10px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Example expressions
EXAMPLES = {
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


def main():
    # Session state
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = np.random.randint(0, 10000)

    st.markdown("## 📐 Inequality Explorer")

    # ===== SIDEBAR =====
    st.sidebar.header("Configuration")

    num_vars = st.sidebar.radio("Variables", [2, 3], horizontal=True)
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

    # Example selection
    st.sidebar.markdown("---")
    example_key = '2 vars' if num_vars == 2 else '3 vars'
    selected_example = st.sidebar.selectbox("Load Example", list(EXAMPLES[example_key].keys()))
    default_latex, default_constraint_latex, default_python, default_constraint_python = EXAMPLES[example_key][selected_example]

    input_mode = st.sidebar.radio("Input Mode", ['Visual (LaTeX)', 'Text (Python)'], horizontal=True)
    find_type = st.sidebar.radio("Find", ['Minimum', 'Maximum'], horizontal=True)

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

        func_initial = default_latex if default_latex else "a^{2}+b^{2}"
        st.markdown(f"**f({', '.join(var_names)}) =**")
        func_result = mathfield(value=func_initial, title="", upright=False, mathml_preview=False)

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

        constraint_initial = default_constraint_latex or ""
        constraint_latex = st.text_input("Constraint g = 0 (optional)", value=constraint_initial,
                                         placeholder="e.g., a + b - 2")
        constraint_expr = LaTeXParser.clean_latex_to_sympy(constraint_latex) if constraint_latex else ""

    else:
        func_expr = st.text_input(f"f({', '.join(var_names)}) =", value=default_python,
                                  placeholder="e.g., a**2 + b**2")
        constraint_expr = st.text_input("Constraint g = 0 (optional)", value=default_constraint_python,
                                        placeholder="e.g., a + b - 2")

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
            if constraint_expr and constraint_expr.strip():
                constraint_func = SafeExpressionParser.create_function(constraint_expr, var_names)

            engine = OptimizationEngine(objective_func, constraint_func, num_vars, bounds)

            if start_point is None:
                start_point = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
                if constraint_func:
                    best_start, best_violation = start_point.copy(), abs(constraint_func(*start_point))
                    for _ in range(100):
                        test = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
                        violation = abs(constraint_func(*test))
                        if violation < best_violation:
                            best_violation, best_start = violation, test
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
            if num_vars == 2:
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
            st.markdown(f"""
            **Problem:** Find the {find_type.lower()} of f({vars_str}) = {func_expr}
            {"**Subject to:** " + constraint_expr + " = 0" if constraint_expr else ""}

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
