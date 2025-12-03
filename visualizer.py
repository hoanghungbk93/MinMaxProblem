"""
Visualization Module
===================
Create interactive visualizations for optimization.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Callable, List, Tuple


class Visualizer:
    """Create interactive visualizations for optimization."""

    @staticmethod
    def create_2d_plot(func: Callable, x_range: Tuple[float, float],
                       resolution: int = 100, var_name: str = 'a') -> go.Figure:
        """Create 2D line plot for single variable function."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.zeros_like(x)

        for i in range(len(x)):
            try:
                y[i] = func(x[i])
            except:
                y[i] = np.nan

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'f({var_name})'
        ))

        fig.update_layout(
            title=f'Function f({var_name})',
            xaxis_title=var_name,
            yaxis_title=f'f({var_name})',
            height=450,
            showlegend=True
        )

        return fig

    @staticmethod
    def add_optimization_path_2d(fig: go.Figure, path: List[np.ndarray],
                                  func: Callable, var_name: str = 'a') -> go.Figure:
        """Add optimization path to 2D plot."""
        if len(path) < 2:
            return fig

        path_x = [p[0] for p in path]
        path_y = []
        for p in path:
            try:
                path_y.append(func(p[0]))
            except:
                path_y.append(np.nan)

        # Add path points
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(size=6, color=list(range(len(path))),
                       colorscale='Reds', showscale=False),
            name='Optimization Path'
        ))

        # Start point
        fig.add_trace(go.Scatter(
            x=[path_x[0]], y=[path_y[0]],
            mode='markers+text',
            marker=dict(size=14, color='limegreen', symbol='circle',
                       line=dict(color='darkgreen', width=2)),
            text=['START'], textposition='top center',
            textfont=dict(size=11, color='darkgreen'),
            name='Start Point'
        ))

        # Optimal point
        fig.add_trace(go.Scatter(
            x=[path_x[-1]], y=[path_y[-1]],
            mode='markers+text',
            marker=dict(size=16, color='gold', symbol='star',
                       line=dict(color='darkorange', width=2)),
            text=['OPTIMAL'], textposition='bottom center',
            textfont=dict(size=12, color='darkorange'),
            name='Optimal Point'
        ))

        return fig

    @staticmethod
    def create_1d_contour_animation(func: Callable, x_range: Tuple[float, float],
                                     path: List[np.ndarray], resolution: int = 100,
                                     var_name: str = 'a', animation_speed: int = 500) -> go.Figure:
        """Create animated 1D plot showing optimization path with gradient coloring."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.zeros_like(x)

        for i in range(len(x)):
            try:
                y[i] = func(x[i])
            except:
                y[i] = np.nan

        path_array = np.array(path)
        n_frames = len(path_array)

        # Compute f values for path
        f_values = []
        for pt in path_array:
            try:
                f_values.append(func(pt[0]))
            except:
                f_values.append(np.nan)

        fig = go.Figure()

        # Base function curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'f({var_name})'
        ))

        # Start point
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[f_values[0]],
            mode='markers+text',
            marker=dict(size=14, color='limegreen', symbol='circle',
                       line=dict(color='darkgreen', width=2)),
            text=['START'], textposition='top center',
            textfont=dict(size=11, color='darkgreen'),
            name='Start'
        ))

        # Optimal point
        fig.add_trace(go.Scatter(
            x=[path_array[-1, 0]], y=[f_values[-1]],
            mode='markers+text',
            marker=dict(size=16, color='gold', symbol='star',
                       line=dict(color='darkorange', width=2)),
            text=['OPTIMAL'], textposition='bottom center',
            textfont=dict(size=12, color='darkorange'),
            name='Optimal'
        ))

        # Path line (animated)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[f_values[0]],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            name='Path'
        ))

        # Current position (animated)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[f_values[0]],
            mode='markers',
            marker=dict(size=14, color='red', symbol='circle',
                       line=dict(color='darkred', width=2)),
            name='Current'
        ))

        # Create frames
        frames = []
        for k in range(1, n_frames + 1):
            curr_x = path_array[k-1, 0]
            curr_f = f_values[k-1]

            frame_data = [
                # Base curve
                go.Scatter(x=x, y=y, mode='lines', line=dict(color='blue', width=2), name=f'f({var_name})'),
                # Start
                go.Scatter(x=[path_array[0, 0]], y=[f_values[0]], mode='markers+text',
                          marker=dict(size=14, color='limegreen', symbol='circle', line=dict(color='darkgreen', width=2)),
                          text=['START'], textposition='top center', textfont=dict(size=11, color='darkgreen'), name='Start'),
                # Optimal
                go.Scatter(x=[path_array[-1, 0]], y=[f_values[-1]], mode='markers+text',
                          marker=dict(size=16, color='gold', symbol='star', line=dict(color='darkorange', width=2)),
                          text=['OPTIMAL'], textposition='bottom center', textfont=dict(size=12, color='darkorange'), name='Optimal'),
                # Path (growing)
                go.Scatter(x=[p[0] for p in path_array[:k]], y=f_values[:k], mode='lines+markers',
                          line=dict(color='red', width=3), marker=dict(size=6, color='red'), name='Path'),
                # Current position
                go.Scatter(x=[curr_x], y=[curr_f], mode='markers',
                          marker=dict(size=14, color='red', symbol='circle', line=dict(color='darkred', width=2)), name='Current')
            ]

            frame_title = f"Step {k}/{n_frames} | {var_name}={curr_x:.4f}, f={curr_f:.4f}"
            frames.append(go.Frame(data=frame_data, name=str(k),
                                  layout=go.Layout(title=dict(text=frame_title, font=dict(size=14)))))

        fig.frames = frames

        # Initial title
        init_title = f"Step 1/{n_frames} | {var_name}={path_array[0, 0]:.4f}, f={f_values[0]:.4f}"

        fig.update_layout(
            title=dict(text=init_title, font=dict(size=14)),
            xaxis_title=var_name,
            yaxis_title=f'f({var_name})',
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.15, 'x': 0.5, 'xanchor': 'center',
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': animation_speed, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 0}}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top', 'xanchor': 'left',
                'currentvalue': {'prefix': 'Step: ', 'visible': True, 'xanchor': 'center'},
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9, 'x': 0.05, 'y': 0,
                'steps': [{'args': [[str(k)], {'frame': {'duration': 0, 'redraw': True},
                                              'mode': 'immediate', 'transition': {'duration': 0}}],
                          'label': str(k), 'method': 'animate'} for k in range(1, n_frames + 1)]
            }]
        )

        return fig

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
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_min = np.percentile(valid_z, 5)   # Use 5th percentile as min
            z_max = np.percentile(valid_z, 95)  # Use 95th percentile as max
            # Add some padding
            z_range = z_max - z_min
            z_min_clip = z_min - 0.1 * z_range
            z_max_clip = z_max + 0.5 * z_range
            Z = np.clip(Z, z_min_clip, z_max_clip)

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
        """Add optimization path to 3D surface plot."""
        if len(path) < 2:
            return fig

        path_array = np.array(path)
        z_path = []
        for p in path:
            try:
                z_path.append(func(p[0], p[1]))
            except:
                z_path.append(np.nan)

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

        fig.add_trace(go.Scatter3d(
            x=[path_array[0, 0]],
            y=[path_array[0, 1]],
            z=[z_path[0]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='Start Point'
        ))

        fig.add_trace(go.Scatter3d(
            x=[path_array[-1, 0]],
            y=[path_array[-1, 1]],
            z=[z_path[-1]],
            mode='markers',
            marker=dict(size=14, color='gold', symbol='diamond',
                       line=dict(color='black', width=2)),
            name='Optimal Point'
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
        """Create contour plot with trajectory visualization."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_max = np.percentile(valid_z, 95)
            Z = np.clip(Z, None, z_max * 1.5)

        fig = go.Figure()

        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            name='f({}, {})'.format(*var_names),
            opacity=0.9,
            colorbar=dict(title='f(a,b)', x=1.02)
        ))

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

        if path is not None and len(path) >= 2:
            path_array = np.array(path)
            n_points = len(path_array)

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

            if show_arrows:
                arrow_interval = max(1, n_points // 10)
                for i in range(0, n_points - 1, arrow_interval):
                    dx = path_array[i+1, 0] - path_array[i, 0]
                    dy = path_array[i+1, 1] - path_array[i, 1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0.001:
                        scale = min(0.15, length * 0.5)
                        dx_norm = dx / length * scale
                        dy_norm = dy / length * scale
                        ax = path_array[i, 0] + dx * 0.5
                        ay = path_array[i, 1] + dy * 0.5
                        fig.add_annotation(
                            x=ax + dx_norm, y=ay + dy_norm,
                            ax=ax, ay=ay,
                            xref='x', yref='y', axref='x', ayref='y',
                            showarrow=True, arrowhead=2, arrowsize=1.5,
                            arrowwidth=2, arrowcolor='rgba(255, 100, 100, 0.8)'
                        )

            sizes = np.linspace(6, 3, n_points)
            fig.add_trace(go.Scatter(
                x=path_array[:, 0], y=path_array[:, 1],
                mode='markers',
                marker=dict(size=sizes, color=list(range(n_points)),
                           colorscale='RdYlBu_r', showscale=False,
                           line=dict(color='white', width=1)),
                name='Path Points',
                hovertemplate='Step %{marker.color}<br>a=%{x:.4f}<br>b=%{y:.4f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=[path_array[0, 0]], y=[path_array[0, 1]],
                mode='markers+text',
                marker=dict(size=18, color='limegreen', symbol='circle',
                           line=dict(color='darkgreen', width=3)),
                text=['START'], textposition='top center',
                textfont=dict(size=12, color='darkgreen', family='Arial Black'),
                name='Start Point'
            ))

            fig.add_trace(go.Scatter(
                x=[path_array[-1, 0]], y=[path_array[-1, 1]],
                mode='markers+text',
                marker=dict(size=22, color='gold', symbol='star',
                           line=dict(color='darkorange', width=3)),
                text=['MIN'], textposition='bottom center',
                textfont=dict(size=14, color='darkorange', family='Arial Black'),
                name='Optimal Point'
            ))

        fig.update_layout(
            title=dict(text='Optimization Trajectory', font=dict(size=16)),
            xaxis_title=var_names[0], yaxis_title=var_names[1],
            height=600, showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                       bgcolor='rgba(255,255,255,0.8)'),
            hovermode='closest'
        )

        return fig

    @staticmethod
    def create_convergence_plot(f_values: List[float], learning_rate: float) -> go.Figure:
        """Create convergence plot."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(f_values))), y=f_values,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            name='f(a,b)'
        ))

        fig.add_hline(y=f_values[-1], line_dash="dash", line_color="red",
                     annotation_text=f"Final: {f_values[-1]:.4f}")

        fig.update_layout(
            title=f'Convergence Plot (LR={learning_rate})',
            xaxis_title='Iteration', yaxis_title='f(a,b)',
            height=300, showlegend=True
        )

        return fig

    @staticmethod
    def create_autoplay_animation(func: Callable, x_range: Tuple[float, float],
                                   y_range: Tuple[float, float],
                                   path: List[np.ndarray],
                                   resolution: int = 50,
                                   var_names: List[str] = ['a', 'b'],
                                   constraint_func: Callable = None,
                                   animation_speed: int = 500) -> go.Figure:
        """Create Plotly animation with frames - click Play to animate step by step.

        Uses Plotly frames with redraw=False to only update path and current point traces,
        keeping the contour surface static for smooth animation.
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_max = np.percentile(valid_z, 95)
            Z = np.clip(Z, None, z_max * 1.5)

        path_array = np.array(path)
        n_frames = len(path_array)

        fig = go.Figure()

        # Trace 0: Contour surface (STATIC - never updated in frames)
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z, colorscale='Viridis', showscale=True, opacity=0.8,
            name='Surface', hoverinfo='skip',
            colorbar=dict(title='f(a,b)', x=1.02)
        ))

        # Trace 1: Constraint if present (STATIC)
        constraint_trace_offset = 0
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
                line=dict(color='cyan', width=4, dash='dash'),
                showscale=False, name='Constraint'
            ))
            constraint_trace_offset = 1

        # Trace 2 (or 1): START point (STATIC)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[path_array[0, 1]],
            mode='markers+text',
            marker=dict(size=18, color='limegreen', symbol='circle',
                       line=dict(color='darkgreen', width=3)),
            text=['START'], textposition='top center',
            textfont=dict(size=12, color='darkgreen', family='Arial Black'),
            name='Start'
        ))

        # Trace 3 (or 2): MIN point (STATIC)
        fig.add_trace(go.Scatter(
            x=[path_array[-1, 0]], y=[path_array[-1, 1]],
            mode='markers+text',
            marker=dict(size=22, color='gold', symbol='star',
                       line=dict(color='darkorange', width=3)),
            text=['MIN'], textposition='bottom center',
            textfont=dict(size=14, color='darkorange', family='Arial Black'),
            name='Minimum'
        ))

        # Trace 4 (or 3): Path line (ANIMATED - starts at first point only)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[path_array[0, 1]],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            name='Path'
        ))

        # Trace 5 (or 4): Current position marker (ANIMATED)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[path_array[0, 1]],
            mode='markers',
            marker=dict(size=16, color='red', symbol='circle',
                       line=dict(color='darkred', width=3)),
            name='Current'
        ))

        # Calculate trace indices for animation
        path_trace_idx = 3 + constraint_trace_offset  # Path trace
        current_trace_idx = 4 + constraint_trace_offset  # Current position trace

        # Compute f values for each point in path
        f_values = []
        for pt in path_array:
            try:
                f_val = func(pt[0], pt[1])
                f_values.append(f_val)
            except:
                f_values.append(np.nan)

        # Create frames - include ALL traces to prevent contour from disappearing
        frames = []
        for k in range(1, n_frames + 1):
            # Current point values
            curr_a = path_array[k-1, 0]
            curr_b = path_array[k-1, 1]
            curr_f = f_values[k-1]

            frame_data = [
                # Contour surface (must be included)
                go.Contour(
                    x=x, y=y, z=Z, colorscale='Viridis', showscale=True, opacity=0.8,
                    name='Surface', hoverinfo='skip',
                    colorbar=dict(title='f(a,b)', x=1.02)
                )
            ]

            # Add constraint if present
            if constraint_func is not None:
                frame_data.append(go.Contour(
                    x=x, y=y, z=C,
                    contours=dict(start=0, end=0, size=0.01, coloring='lines'),
                    line=dict(color='cyan', width=4, dash='dash'),
                    showscale=False, name='Constraint'
                ))

            # START point
            frame_data.append(go.Scatter(
                x=[path_array[0, 0]], y=[path_array[0, 1]],
                mode='markers+text',
                marker=dict(size=18, color='limegreen', symbol='circle',
                           line=dict(color='darkgreen', width=3)),
                text=['START'], textposition='top center',
                textfont=dict(size=12, color='darkgreen', family='Arial Black'),
                name='Start'
            ))

            # MIN point
            frame_data.append(go.Scatter(
                x=[path_array[-1, 0]], y=[path_array[-1, 1]],
                mode='markers+text',
                marker=dict(size=22, color='gold', symbol='star',
                           line=dict(color='darkorange', width=3)),
                text=['MIN'], textposition='bottom center',
                textfont=dict(size=14, color='darkorange', family='Arial Black'),
                name='Minimum'
            ))

            # Path line (growing)
            frame_data.append(go.Scatter(
                x=path_array[:k, 0].tolist(),
                y=path_array[:k, 1].tolist(),
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Path'
            ))

            # Current position marker
            frame_data.append(go.Scatter(
                x=[path_array[k-1, 0]],
                y=[path_array[k-1, 1]],
                mode='markers',
                marker=dict(size=16, color='red', symbol='circle',
                           line=dict(color='darkred', width=3)),
                name='Current'
            ))

            # Create frame with layout update for title
            frame_title = f"Step {k}/{n_frames} | a={curr_a:.4f}, b={curr_b:.4f}, f={curr_f:.4f}"
            frames.append(go.Frame(
                data=frame_data,
                name=str(k),
                layout=go.Layout(title=dict(text=frame_title, font=dict(size=14)))
            ))

        fig.frames = frames

        # Store C for use in frames if constraint exists
        if constraint_func is not None:
            C = C  # Already computed above

        # Initial title with first point values
        init_a, init_b, init_f = path_array[0, 0], path_array[0, 1], f_values[0]
        init_title = f"Step 1/{n_frames} | a={init_a:.4f}, b={init_b:.4f}, f={init_f:.4f}"

        # Layout with Play/Pause buttons and slider
        fig.update_layout(
            title=dict(text=init_title, font=dict(size=14)),
            xaxis_title=var_names[0], yaxis_title=var_names[1],
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='closest',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.15, 'x': 0.5, 'xanchor': 'center',
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': animation_speed, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top', 'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Step: ',
                    'visible': True,
                    'xanchor': 'center'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9, 'x': 0.05, 'y': 0,
                'steps': [{
                    'args': [[str(k)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(k),
                    'method': 'animate'
                } for k in range(1, n_frames + 1)]
            }]
        )

        return fig

    @staticmethod
    def create_static_contour_base(func: Callable, x_range: Tuple[float, float],
                                    y_range: Tuple[float, float],
                                    resolution: int = 50,
                                    var_names: List[str] = ['a', 'b'],
                                    constraint_func: Callable = None) -> go.Figure:
        """Create the static contour base (computed once, reused for animation)."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_max = np.percentile(valid_z, 95)
            Z = np.clip(Z, None, z_max * 1.5)

        fig = go.Figure()

        # Contour surface
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z, colorscale='Viridis', showscale=True, opacity=0.8,
            name='Surface', hoverinfo='skip'
        ))

        # Add constraint if present
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
                line=dict(color='cyan', width=4, dash='dash'),
                showscale=False, name='Constraint'
            ))

        fig.update_layout(
            xaxis_title=var_names[0], yaxis_title=var_names[1],
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)')
        )

        return fig

    @staticmethod
    def create_animation_frame(func: Callable, x_range: Tuple[float, float],
                                y_range: Tuple[float, float],
                                path: List[np.ndarray],
                                current_step: int,
                                resolution: int = 50,
                                var_names: List[str] = ['a', 'b'],
                                constraint_func: Callable = None) -> go.Figure:
        """Create a single frame for the animation at the given step."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_max = np.percentile(valid_z, 95)
            Z = np.clip(Z, None, z_max * 1.5)

        path_array = np.array(path)
        n_total = len(path_array)

        fig = go.Figure()

        # Contour surface
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z, colorscale='Viridis', showscale=True, opacity=0.8,
            name='Surface', hoverinfo='skip'
        ))

        # Add constraint if present
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
                line=dict(color='cyan', width=4, dash='dash'),
                showscale=False, name='Constraint'
            ))

        # Start point (always visible)
        fig.add_trace(go.Scatter(
            x=[path_array[0, 0]], y=[path_array[0, 1]],
            mode='markers+text',
            marker=dict(size=18, color='limegreen', symbol='circle',
                       line=dict(color='darkgreen', width=3)),
            text=['START'], textposition='top center',
            textfont=dict(size=11, color='darkgreen'),
            name='Start'
        ))

        # Minimum point (always visible)
        fig.add_trace(go.Scatter(
            x=[path_array[-1, 0]], y=[path_array[-1, 1]],
            mode='markers+text',
            marker=dict(size=20, color='gold', symbol='star',
                       line=dict(color='darkorange', width=3)),
            text=['MIN'], textposition='bottom center',
            textfont=dict(size=12, color='darkorange'),
            name='Minimum'
        ))

        # Path traveled so far
        if current_step > 0:
            fig.add_trace(go.Scatter(
                x=path_array[:current_step, 0], y=path_array[:current_step, 1],
                mode='lines',
                line=dict(color='red', width=3),
                name='Path'
            ))

        # Current position marker
        if current_step > 0:
            fig.add_trace(go.Scatter(
                x=[path_array[current_step-1, 0]], y=[path_array[current_step-1, 1]],
                mode='markers',
                marker=dict(size=14, color='red', symbol='circle',
                           line=dict(color='darkred', width=2)),
                name='Current'
            ))

        fig.update_layout(
            title=dict(text=f'Step {current_step}/{n_total}', font=dict(size=14)),
            xaxis_title=var_names[0], yaxis_title=var_names[1],
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)')
        )

        return fig

    @staticmethod
    def create_animated_contour(func: Callable, x_range: Tuple[float, float],
                                 y_range: Tuple[float, float],
                                 path: List[np.ndarray],
                                 resolution: int = 50,
                                 var_names: List[str] = ['a', 'b'],
                                 constraint_func: Callable = None,
                                 animation_speed: int = 100) -> go.Figure:
        """Create animated contour plot that auto-plays from start to minimum."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = func(X[i, j], Y[i, j])
                    # Handle infinity and very large values
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e6:
                        Z[i, j] = np.nan
                    else:
                        Z[i, j] = val
                except:
                    Z[i, j] = np.nan

        # Clip z values to reasonable range for visualization
        valid_z = Z[~np.isnan(Z)]
        if len(valid_z) > 0:
            z_max = np.percentile(valid_z, 95)
            Z = np.clip(Z, None, z_max * 1.5)

        path_array = np.array(path)
        n_frames = len(path_array)

        # Create frames for animation
        frames = []
        for k in range(1, n_frames + 1):
            frame_data = [
                go.Contour(x=x, y=y, z=Z, colorscale='Viridis', showscale=False, opacity=0.8,
                          name='Surface', hoverinfo='skip'),
                # Start point (always visible)
                go.Scatter(
                    x=[path_array[0, 0]], y=[path_array[0, 1]],
                    mode='markers+text',
                    marker=dict(size=18, color='limegreen', symbol='circle',
                               line=dict(color='darkgreen', width=3)),
                    text=['START'], textposition='top center',
                    textfont=dict(size=11, color='darkgreen'),
                    name='Start', showlegend=True
                ),
                # Minimum point (always visible)
                go.Scatter(
                    x=[path_array[-1, 0]], y=[path_array[-1, 1]],
                    mode='markers+text',
                    marker=dict(size=20, color='gold', symbol='star',
                               line=dict(color='darkorange', width=3)),
                    text=['MIN'], textposition='bottom center',
                    textfont=dict(size=12, color='darkorange'),
                    name='Minimum', showlegend=True
                ),
                # Path traveled so far
                go.Scatter(
                    x=path_array[:k, 0], y=path_array[:k, 1],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Path', showlegend=(k == 1)
                ),
                # Current position marker
                go.Scatter(
                    x=[path_array[k-1, 0]], y=[path_array[k-1, 1]],
                    mode='markers',
                    marker=dict(size=14, color='red', symbol='circle',
                               line=dict(color='darkred', width=2)),
                    name='Current', showlegend=True
                )
            ]
            frames.append(go.Frame(data=frame_data, name=str(k)))

        # Initial figure - show start and minimum points
        fig = go.Figure(
            data=[
                go.Contour(x=x, y=y, z=Z, colorscale='Viridis', showscale=True, opacity=0.8,
                          name='Surface', hoverinfo='skip'),
                # Start point
                go.Scatter(
                    x=[path_array[0, 0]], y=[path_array[0, 1]],
                    mode='markers+text',
                    marker=dict(size=18, color='limegreen', symbol='circle',
                               line=dict(color='darkgreen', width=3)),
                    text=['START'], textposition='top center',
                    textfont=dict(size=11, color='darkgreen'),
                    name='Start'
                ),
                # Minimum point
                go.Scatter(
                    x=[path_array[-1, 0]], y=[path_array[-1, 1]],
                    mode='markers+text',
                    marker=dict(size=20, color='gold', symbol='star',
                               line=dict(color='darkorange', width=3)),
                    text=['MIN'], textposition='bottom center',
                    textfont=dict(size=12, color='darkorange'),
                    name='Minimum'
                ),
                # Empty path (will be filled by animation)
                go.Scatter(x=[], y=[], mode='lines', line=dict(color='red', width=3), name='Path'),
                # Current position (starts at start point)
                go.Scatter(
                    x=[path_array[0, 0]], y=[path_array[0, 1]],
                    mode='markers',
                    marker=dict(size=14, color='red', symbol='circle',
                               line=dict(color='darkred', width=2)),
                    name='Current'
                )
            ],
            frames=frames
        )

        # Add constraint if present
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
                line=dict(color='cyan', width=4, dash='dash'),
                showscale=False, name='Constraint'
            ))

        # Auto-play animation layout
        fig.update_layout(
            title=dict(text=f'Optimization: Start → Minimum ({n_frames} steps)', font=dict(size=14)),
            xaxis_title=var_names[0], yaxis_title=var_names[1],
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
            sliders=[{
                'active': 0,
                'steps': [{'args': [[str(k)], {'frame': {'duration': 0, 'redraw': True},
                                              'mode': 'immediate', 'transition': {'duration': 0}}],
                          'label': str(k), 'method': 'animate'} for k in range(1, n_frames + 1)],
                'x': 0.1, 'len': 0.8, 'y': -0.02,
                'currentvalue': {'prefix': 'Step: ', 'visible': True, 'xanchor': 'center'},
                'transition': {'duration': 0}
            }]
        )

        # Add auto-play JavaScript via config
        fig.layout.updatemenus = [{
            'type': 'buttons',
            'showactive': False,
            'y': 1.12, 'x': 0.5, 'xanchor': 'center',
            'buttons': [
                {'label': '▶ Replay', 'method': 'animate',
                 'args': [None, {'frame': {'duration': animation_speed, 'redraw': True},
                                'fromcurrent': False, 'mode': 'immediate',
                                'transition': {'duration': 0}}]},
                {'label': '⏸ Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }]

        return fig
