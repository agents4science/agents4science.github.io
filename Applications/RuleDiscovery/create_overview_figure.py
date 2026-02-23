#!/usr/bin/env python3
"""
Create illustrative overview figure for Rule Discovery.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


def create_rule_discovery_figure():
    """Create illustrative figure for Rule Discovery."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, 'Rule Discovery', fontsize=22, fontweight='bold',
            ha='center', va='top', color='#2c3e50')
    ax.text(6, 6.35, 'LLM agents discover hidden rules through experimentation',
            fontsize=12, ha='center', va='top', style='italic', color='#555')

    # Hidden rules box (top left)
    rules_box = FancyBboxPatch((0.2, 4.3), 3.6, 1.7, boxstyle="round,pad=0.05",
                                facecolor='#34495e', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rules_box)
    ax.text(2.0, 5.75, 'HIDDEN RULES', fontsize=10, ha='center', va='center',
            color='#bdc3c7', fontweight='bold')
    rules_text = [
        'Metal conducts electricity',
        'Glass shatters when dropped',
        'Wood burns in fire',
    ]
    for i, rule in enumerate(rules_text):
        ax.text(2.0, 5.35 - i*0.35, f'• {rule}', fontsize=9, ha='center',
                color='white')

    # World simulator (center top)
    world_box = FancyBboxPatch((4.2, 4.3), 3.6, 1.7, boxstyle="round,pad=0.05",
                                facecolor='#8e44ad', edgecolor='#6c3483', linewidth=2)
    ax.add_patch(world_box)
    ax.text(6.0, 5.75, 'WORLD SIMULATOR', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')

    # Draw objects in world
    obj_colors = ['#e74c3c', '#3498db', '#f1c40f', '#95a5a6']
    obj_x = [4.8, 5.4, 6.0, 6.6, 7.2]
    for i, ox in enumerate(obj_x):
        # Small colored squares representing objects
        obj = FancyBboxPatch((ox-0.15, 5.1), 0.3, 0.3, boxstyle="round,pad=0.02",
                              facecolor=obj_colors[i % len(obj_colors)],
                              edgecolor='white', linewidth=1)
        ax.add_patch(obj)

    ax.text(6.0, 4.7, 'water | fire | electricity | drop', fontsize=8, ha='center',
            color='#d7bde2', style='italic')

    # Draw 3 LLM agents
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71']
    agent_x = [2, 6, 10]

    for i, (x, color) in enumerate(zip(agent_x, agent_colors)):
        # Agent box (representing LLM)
        agent_box = FancyBboxPatch((x-0.9, 2.0), 1.8, 1.2, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='white', linewidth=3)
        ax.add_patch(agent_box)
        ax.text(x, 2.85, f'Agent {i+1}', fontsize=11, ha='center', va='center',
                color='white', fontweight='bold')
        ax.text(x, 2.4, '[ LLM ]', fontsize=9, ha='center', va='center',
                color='white', alpha=0.8)

        # Experiment bubble
        exp_texts = [
            '"Drop glass\ninto water"',
            '"Apply electricity\nto metal cube"',
            '"Expose wood\nto fire"'
        ]
        bubble = FancyBboxPatch((x-0.8, 1.2), 1.6, 0.5, boxstyle="round,pad=0.1",
                                 facecolor='#fef9e7', edgecolor='#f4d03f', linewidth=1.5)
        ax.add_patch(bubble)
        ax.text(x, 1.45, exp_texts[i], fontsize=8, ha='center', va='center',
               style='italic', color='#7d6608')

    # Arrows from agents to world (experiments)
    ax.annotate('', xy=(5.0, 4.2), xytext=(2.5, 3.3),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2,
                              connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(6.0, 4.2), xytext=(6.0, 3.3),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.annotate('', xy=(7.0, 4.2), xytext=(9.5, 3.3),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2,
                              connectionstyle='arc3,rad=-0.2'))

    ax.text(4.3, 3.4, 'run\nexperiments', fontsize=8, ha='center', color='#7f8c8d',
            style='italic')

    # Arrows from world back to agents (observations) - dashed
    ax.annotate('', xy=(2.5, 3.3), xytext=(4.5, 4.2),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5,
                              connectionstyle='arc3,rad=0.2', linestyle='--'))

    ax.text(2.8, 3.8, 'observe\noutcomes', fontsize=8, ha='center', color='#27ae60',
            style='italic')

    # Communication arrows between agents
    ax.annotate('', xy=(5.0, 2.5), xytext=(3.0, 2.5),
               arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=2.5))
    ax.annotate('', xy=(9.0, 2.5), xytext=(7.0, 2.5),
               arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=2.5))

    ax.text(4.0, 2.9, 'share', fontsize=9, ha='center', va='bottom',
            color='#e67e22', fontweight='bold')
    ax.text(8.0, 2.9, 'share', fontsize=9, ha='center', va='bottom',
            color='#e67e22', fontweight='bold')

    # Discovered rules box (right side)
    result_box = FancyBboxPatch((8.2, 4.3), 3.6, 1.7, boxstyle="round,pad=0.05",
                                 facecolor='#27ae60', edgecolor='#1e8449', linewidth=2)
    ax.add_patch(result_box)
    ax.text(10.0, 5.75, 'DISCOVERED RULES', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    discovered = [
        'Metal conducts (100%)',
        'Glass shatters (100%)',
        'Wood burns (100%)',
    ]
    for i, rule in enumerate(discovered):
        ax.text(10.0, 5.35 - i*0.35, f'✓ {rule}', fontsize=9, ha='center',
                color='#d5f5e3')

    # Arrow from world to results
    ax.annotate('', xy=(8.1, 5.1), xytext=(7.9, 5.1),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=4))

    # Process steps at bottom - showing what each LLM agent does
    # Background box for the workflow
    workflow_box = FancyBboxPatch((0.3, -0.1), 11.4, 0.7, boxstyle="round,pad=0.05",
                                   facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=1.5)
    ax.add_patch(workflow_box)

    # Header label
    ax.text(6, 0.75, 'EACH LLM AGENT LOOP:', fontsize=10, ha='center', va='center',
            color='#2c3e50', fontweight='bold')

    # Steps with arrows between them
    steps = [
        ('1', 'Propose\nexperiment'),
        ('2', 'Run & observe\noutcome'),
        ('3', 'Mine\nhypotheses'),
        ('4', 'Share with\npeers'),
    ]
    step_x = [1.5, 4.2, 7.0, 10.0]

    for i, ((num, text), x) in enumerate(zip(steps, step_x)):
        # Step circle
        circle = Circle((x, 0.35), 0.25, facecolor='#3498db', edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 0.35, num, fontsize=11, ha='center', va='center',
                color='white', fontweight='bold')
        # Step label
        ax.text(x + 0.4, 0.35, text, fontsize=9, ha='left', va='center',
               color='#2c3e50', fontweight='bold')

        # Arrow to next step
        if i < len(steps) - 1:
            next_x = step_x[i + 1]
            ax.annotate('', xy=(next_x - 0.5, 0.35), xytext=(x + 1.6, 0.35),
                       arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5))

    # Loop arrow from step 4 back to step 1
    ax.annotate('', xy=(1.6, 0.15), xytext=(9.85, 0.15),
               arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5,
                              connectionstyle='arc3,rad=-0.05'))

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs('figures', exist_ok=True)

    plt.savefig('figures/overview.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figures/overview.png")
    plt.close()


if __name__ == "__main__":
    create_rule_discovery_figure()
