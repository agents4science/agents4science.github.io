#!/usr/bin/env python3
"""
Create illustrative overview figure for Polynomial Discovery.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


def create_polynomial_discovery_figure():
    """Create illustrative figure for Polynomial Discovery."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, 'Polynomial Discovery', fontsize=22, fontweight='bold',
            ha='center', va='top', color='#2c3e50')
    ax.text(6, 6.15, 'Agents collaboratively learn a hidden function from noisy observations',
            fontsize=12, ha='center', va='top', style='italic', color='#555')

    # Hidden function box (top center)
    hidden_box = FancyBboxPatch((3.8, 4.6), 4.4, 1.3, boxstyle="round,pad=0.05",
                                 facecolor='#34495e', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(6, 5.55, 'HIDDEN FUNCTION', fontsize=11, ha='center', va='center',
            color='#bdc3c7', fontweight='bold')
    ax.text(6, 5.05, 'f(x) = ax³ + bx² + cx + d', fontsize=13, ha='center', va='center',
            color='white', family='monospace', fontweight='bold')

    # Draw 4 agents in a row
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    agent_x = [1.5, 4, 6.5, 9]
    agent_names = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']

    for i, (x, color, name) in enumerate(zip(agent_x, agent_colors, agent_names)):
        # Agent circle
        circle = Circle((x, 2.5), 0.55, facecolor=color, edgecolor='white', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, 2.5, f'A{i+1}', fontsize=14, ha='center', va='center',
                color='white', fontweight='bold')

        # Arrow from hidden function to agent (sampling)
        start_x = 5.2 if i < 2 else 6.8
        ax.annotate('', xy=(x, 3.15), xytext=(start_x, 4.5),
                   arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5,
                                  connectionstyle='arc3,rad=0.15'))

        # Small scatter plot below each agent showing noisy data
        np.random.seed(42 + i)
        xs = np.random.uniform(-1, 1, 10)
        true_y = 0.5*xs**3 - 0.3*xs**2 + 0.8*xs + 0.2
        noisy_y = true_y + np.random.normal(0, 0.2, len(xs))

        # Mini plot area
        plot_left = x - 0.7
        plot_bottom = 0.5
        plot_width = 1.4
        plot_height = 1.1

        # Background for mini plot
        mini_bg = FancyBboxPatch((plot_left - 0.05, plot_bottom - 0.05),
                                  plot_width + 0.1, plot_height + 0.1,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1)
        ax.add_patch(mini_bg)

        # Scale points to mini plot
        px = plot_left + (xs + 1) / 2 * plot_width
        py = plot_bottom + (noisy_y + 1) / 2.5 * plot_height

        ax.scatter(px, py, c=color, s=20, alpha=0.8, zorder=5, edgecolor='white', linewidth=0.5)

        # Fitted line
        x_line = np.linspace(-1, 1, 50)
        y_line = 0.5*x_line**3 - 0.3*x_line**2 + 0.8*x_line + 0.2
        px_line = plot_left + (x_line + 1) / 2 * plot_width
        py_line = plot_bottom + (y_line + 1) / 2.5 * plot_height
        ax.plot(px_line, py_line, color=color, lw=2, alpha=0.6, zorder=4)

    # Communication arrows between agents (bidirectional)
    for i in range(len(agent_x) - 1):
        mid_y = 2.5
        # Top arrow (right)
        ax.annotate('', xy=(agent_x[i+1] - 0.7, mid_y + 0.15),
                   xytext=(agent_x[i] + 0.7, mid_y + 0.15),
                   arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2.5))
        # Bottom arrow (left)
        ax.annotate('', xy=(agent_x[i] + 0.7, mid_y - 0.15),
                   xytext=(agent_x[i+1] - 0.7, mid_y - 0.15),
                   arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2.5))

    # Labels
    ax.text(5.25, 3.0, 'SHARE MODELS', fontsize=10, ha='center', va='bottom',
            color='#e67e22', fontweight='bold')

    ax.text(3, 4.1, 'sample x,\nobserve noisy y', fontsize=9, ha='center', va='top',
            color='#7f8c8d', style='italic')

    # Result arrow
    ax.annotate('', xy=(10.8, 5.2), xytext=(8.3, 5.2),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=4))

    # Result box
    result_box = FancyBboxPatch((10.3, 4.6), 1.5, 1.3, boxstyle="round,pad=0.05",
                                 facecolor='#27ae60', edgecolor='#1e8449', linewidth=2)
    ax.add_patch(result_box)
    ax.text(11.05, 5.45, 'LEARNED', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    ax.text(11.05, 4.95, 'MSE\n< 0.001', fontsize=11, ha='center', va='center',
            color='#d5f5e3', fontweight='bold')

    # Process steps at bottom
    steps = [
        ('1', 'Sample noisy\nobservations'),
        ('2', 'Fit polynomial\nmodel'),
        ('3', 'Share with\npeers'),
        ('4', 'Converge to\ntruth'),
    ]
    step_x = [1.5, 4.0, 7.0, 10.0]
    for (num, text), x in zip(steps, step_x):
        ax.text(x, 0.15, f'{num}. {text}', fontsize=9, ha='center', va='center',
               color='#555', fontweight='bold')

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs('figures', exist_ok=True)

    plt.savefig('figures/overview.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figures/overview.png")
    plt.close()


if __name__ == "__main__":
    create_polynomial_discovery_figure()
