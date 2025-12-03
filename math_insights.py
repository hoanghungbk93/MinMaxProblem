"""
Math Insight Analyzer
====================
Analyze optimization results for mathematical insights.
"""

import numpy as np
from typing import Dict, List, Optional


class MathInsightAnalyzer:
    """Analyze optimization results for mathematical insights."""

    TOLERANCE = 0.05

    @classmethod
    def analyze(cls, x_optimal: np.ndarray, var_names: List[str],
                f_optimal: float, constraint_expr: str = None) -> List[Dict]:
        """Analyze the optimal point and generate insights."""
        insights = []

        # Convert to numpy array and ensure float type
        x_optimal = np.array([float(x) for x in x_optimal])
        f_optimal = float(f_optimal)

        symmetry_insight = cls._check_symmetry(x_optimal, var_names)
        if symmetry_insight:
            insights.append(symmetry_insight)

        boundary_insight = cls._check_boundary(x_optimal, var_names)
        if boundary_insight:
            insights.append(boundary_insight)

        ratio_insights = cls._check_ratios(x_optimal, var_names)
        insights.extend(ratio_insights)

        special_insights = cls._check_special_values(x_optimal, var_names)
        insights.extend(special_insights)

        general_insight = cls._generate_general_suggestion(x_optimal, var_names, f_optimal)
        if general_insight:
            insights.append(general_insight)

        return insights

    @classmethod
    def _check_symmetry(cls, x: np.ndarray, var_names: List[str]) -> Optional[Dict]:
        """Check if variables are approximately equal."""
        if len(x) < 2:
            return None

        mean_val = np.mean(x)
        if mean_val == 0:
            return None

        relative_diffs = np.abs(x - mean_val) / abs(mean_val)

        if np.all(relative_diffs < cls.TOLERANCE):
            vars_str = ' = '.join(var_names)
            return {
                'type': 'symmetry',
                'icon': 'sync',
                'title': 'Symmetry Detected',
                'description': f'All variables are approximately equal: {vars_str} = {mean_val:.4f}',
                'suggestion': 'This suggests using AM-GM inequality or Cauchy-Schwarz.',
                'technique': 'AM-GM / Cauchy-Schwarz'
            }

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if abs(x[i]) > cls.TOLERANCE and abs(x[i] - x[j]) / abs(x[i]) < cls.TOLERANCE:
                    return {
                        'type': 'partial_symmetry',
                        'icon': 'compare_arrows',
                        'title': 'Partial Symmetry',
                        'description': f'{var_names[i]} = {var_names[j]} = {x[i]:.4f}',
                        'suggestion': f'Consider substituting {var_names[j]} = {var_names[i]}.',
                        'technique': 'Variable Substitution'
                    }

        return None

    @classmethod
    def _check_boundary(cls, x: np.ndarray, var_names: List[str]) -> Optional[Dict]:
        """Check if any variable is at boundary."""
        boundary_vars = []

        for i, val in enumerate(x):
            if abs(val) < 0.1:
                boundary_vars.append((var_names[i], val, 'near zero'))
            elif val > 100:
                boundary_vars.append((var_names[i], val, 'very large'))

        if boundary_vars:
            desc_parts = [f'{v[0]} -> {v[2]} ({v[1]:.4f})' for v in boundary_vars]
            return {
                'type': 'boundary',
                'icon': 'location_on',
                'title': 'Boundary Case',
                'description': 'Minimum at boundary: ' + ', '.join(desc_parts),
                'suggestion': 'Use L\'Hopital\'s rule or asymptotic analysis.',
                'technique': 'Boundary Analysis'
            }

        return None

    @classmethod
    def _check_ratios(cls, x: np.ndarray, var_names: List[str]) -> List[Dict]:
        """Check for specific ratios between variables."""
        insights = []

        common_ratios = [
            (2, '2:1'), (3, '3:1'), (0.5, '1:2'), (1/3, '1:3'),
            (np.sqrt(2), 'sqrt(2):1'), (np.sqrt(3), 'sqrt(3):1'),
            ((1 + np.sqrt(5)) / 2, 'phi:1 (Golden Ratio)'),
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
                            'icon': 'straighten',
                            'title': f'Special Ratio: {ratio_name}',
                            'description': f'{var_names[i]} / {var_names[j]} = {ratio:.4f}',
                            'suggestion': f'Try {var_names[i]} = {target_ratio:.4f} * {var_names[j]}',
                            'technique': 'Variable Ratio Substitution'
                        })
                        break

        return insights

    @classmethod
    def _check_special_values(cls, x: np.ndarray, var_names: List[str]) -> List[Dict]:
        """Check for special mathematical values."""
        insights = []

        special_values = [
            (1, '1'), (2, '2'), (np.e, 'e'), (np.pi, 'pi'),
            (np.sqrt(2), 'sqrt(2)'), (np.sqrt(3), 'sqrt(3)'),
            ((1 + np.sqrt(5)) / 2, 'phi (Golden Ratio)'),
        ]

        for i, val in enumerate(x):
            for target, name in special_values:
                if abs(val - target) < cls.TOLERANCE * target:
                    insights.append({
                        'type': 'special_value',
                        'icon': 'auto_awesome',
                        'title': f'Special Value: {name}',
                        'description': f'{var_names[i]} = {name} = {target:.4f}',
                        'suggestion': 'This might be derivable analytically.',
                        'technique': 'Analytical Solution'
                    })
                    break

        return insights

    @classmethod
    def _generate_general_suggestion(cls, x: np.ndarray, var_names: List[str],
                                     f_optimal: float) -> Optional[Dict]:
        """Generate general suggestion based on optimal point."""
        nice_values = [0, 1, 2, 3, 4, 5, 0.5, 0.25, 0.75]

        for nice in nice_values:
            if abs(f_optimal - nice) < cls.TOLERANCE:
                return {
                    'type': 'nice_optimal',
                    'icon': 'gps_fixed',
                    'title': 'Clean Optimal Value',
                    'description': f'f* = {nice} is a "nice" number.',
                    'suggestion': 'The inequality can likely be proven using classical techniques.',
                    'technique': 'Classical Inequalities'
                }

        return None
