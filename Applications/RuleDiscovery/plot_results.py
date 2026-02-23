#!/usr/bin/env python3
"""
Generate visualization graphs for the Rule Discovery experiments.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Data from experiments
experiments = [160, 320, 480, 640, 800]
before_scores = [31, 37, 50, None, None]  # Average of 25-37 for 160
after_scores = [56, 62, 75, 87, 75]  # Average of 50-62 for 160
before_rules = [2.5, 3, 4, None, None]
after_rules = [4.5, 5, 6, 7, 6]

# Figure 1: Before vs After Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Score comparison
ax1 = axes[0]
x = np.arange(3)
width = 0.35
before_vals = [31, 37, 50]
after_vals = [56, 62, 75]

bars1 = ax1.bar(x - width/2, before_vals, width, label='Before Improvements', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, after_vals, width, label='After Improvements', color='#27ae60', alpha=0.8)

ax1.set_ylabel('Score (%)', fontsize=14)
ax1.set_xlabel('Total Experiments', fontsize=14)
ax1.set_title('Hard Difficulty: Before vs After Improvements', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['160', '320', '480'])
ax1.legend(loc='upper left')
ax1.set_ylim(0, 100)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# Right plot: Rules found
ax2 = axes[1]
before_rules_vals = [2.5, 3, 4]
after_rules_vals = [4.5, 5, 6]

bars3 = ax2.bar(x - width/2, before_rules_vals, width, label='Before Improvements', color='#e74c3c', alpha=0.8)
bars4 = ax2.bar(x + width/2, after_rules_vals, width, label='After Improvements', color='#27ae60', alpha=0.8)

ax2.set_ylabel('Rules Found (out of 8)', fontsize=14)
ax2.set_xlabel('Total Experiments', fontsize=14)
ax2.set_title('Hard Difficulty: Rules Discovered', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['160', '320', '480'])
ax2.legend(loc='upper left')
ax2.set_ylim(0, 8)
ax2.axhline(y=8, color='gray', linestyle='--', alpha=0.5, label='All 8 rules')

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('figures/hard_before_after.png', dpi=150, bbox_inches='tight')
print("Saved: figures/hard_before_after.png")

# Figure 2: Scaling with Experiments (After improvements only)
fig, ax = plt.subplots(figsize=(10, 6))

exp_full = [160, 320, 480, 640, 800]
scores_full = [56, 62, 75, 87, 75]
rules_full = [4.5, 5, 6, 7, 6]

ax.plot(exp_full, scores_full, 'o-', linewidth=2.5, markersize=10, color='#3498db', label='Score (%)')
ax.fill_between(exp_full, scores_full, alpha=0.2, color='#3498db')

ax.set_xlabel('Total Experiments', fontsize=14)
ax.set_ylabel('Score (%)', fontsize=14)
ax.set_title('Hard Difficulty: Score vs Experiment Count\n(After Improvements)', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_xlim(100, 850)

# Add annotations for rules found
for i, (e, s, r) in enumerate(zip(exp_full, scores_full, rules_full)):
    ax.annotate(f'{s}%\n({r:.0f}/8 rules)', xy=(e, s), xytext=(0, 15),
               textcoords="offset points", ha='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(850, 100, '100% (all rules)', va='center', ha='right', color='green', fontsize=11)

plt.tight_layout()
plt.savefig('figures/hard_scaling.png', dpi=150, bbox_inches='tight')
print("Saved: figures/hard_scaling.png")

# Figure 3: Communication Impact (from earlier vary_agents results)
fig, ax = plt.subplots(figsize=(10, 6))

agent_counts = [1, 2, 4, 8, 16]
no_comm_scores = [20, 40, 20, 40, 40]
n1e2_scores = [20, 40, 20, 60, 40]

x = np.arange(len(agent_counts))
width = 0.35

bars1 = ax.bar(x - width/2, no_comm_scores, width, label='No Communication', color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x + width/2, n1e2_scores, width, label='N1_E2 (1 peer every 2 exp)', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Score (%)', fontsize=14)
ax.set_xlabel('Number of Agents', fontsize=14)
ax.set_title('Medium Difficulty: Impact of Communication\n(20 experiments per agent)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(agent_counts)
ax.legend(loc='upper left')
ax.set_ylim(0, 100)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/communication_impact.png', dpi=150, bbox_inches='tight')
print("Saved: figures/communication_impact.png")

# Figure 4: Multi-seed variance (from multi_seed_results)
fig, ax = plt.subplots(figsize=(10, 6))

seeds = ['Seed 42', 'Seed 123', 'Seed 456', 'Seed 789', 'Seed 1001']
no_comm_multi = [33, 33, 33, 33, 33]
n1e2_multi = [33, 33, 66, 33, 66]

x = np.arange(len(seeds))
width = 0.35

bars1 = ax.bar(x - width/2, no_comm_multi, width, label='No Communication', color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x + width/2, n1e2_multi, width, label='N1_E2 (with communication)', color='#9b59b6', alpha=0.8)

# Add mean lines
ax.axhline(y=np.mean(no_comm_multi), color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2)
ax.axhline(y=np.mean(n1e2_multi), color='#8e44ad', linestyle='--', alpha=0.7, linewidth=2)

ax.text(4.6, np.mean(no_comm_multi), f'Mean: {np.mean(no_comm_multi):.0f}%', va='center', color='#7f8c8d', fontsize=11)
ax.text(4.6, np.mean(n1e2_multi), f'Mean: {np.mean(n1e2_multi):.0f}%', va='center', color='#8e44ad', fontsize=11)

ax.set_ylabel('Score (%)', fontsize=14)
ax.set_xlabel('Random Seed', fontsize=14)
ax.set_title('Easy Difficulty: Variance Across Seeds\n(4 agents, 60 experiments each)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.legend(loc='upper right')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('figures/seed_variance.png', dpi=150, bbox_inches='tight')
print("Saved: figures/seed_variance.png")

# Figure 5: Summary dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Difficulty comparison
ax = axes[0, 0]
difficulties = ['Easy', 'Medium', 'Hard\n(before)', 'Hard\n(after)']
best_scores = [66, 80, 50, 87]
colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#27ae60']

bars = ax.bar(difficulties, best_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Best Score (%)', fontsize=14)
ax.set_title('Best Scores by Difficulty', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

for bar, score in zip(bars, best_scores):
    ax.annotate(f'{score}%', xy=(bar.get_x() + bar.get_width()/2, score),
               xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

# Top right: Improvement breakdown
ax = axes[0, 1]
improvements = ['Scale action\nfix', 'Forced\ndiversity', 'Lower\nthreshold', 'Targeted\nfollow-up']
impact = [12.5, 15, 10, 5]  # Estimated impact percentages

bars = ax.barh(improvements, impact, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Estimated Impact (%)', fontsize=14)
ax.set_title('Improvement Contributions', fontsize=16, fontweight='bold')
ax.set_xlim(0, 25)

for bar, val in zip(bars, impact):
    ax.annotate(f'+{val}%', xy=(val, bar.get_y() + bar.get_height()/2),
               xytext=(5, 0), textcoords="offset points", va='center', fontsize=11)

# Bottom left: Rules by property type
ax = axes[1, 0]
rule_types = ['Material\n(metal, wood)', 'Shape\n(sphere, pyramid)', 'Size\n(small)', 'Color\n(red)']
found_before = [2, 1, 0, 0]
found_after = [4, 2, 1, 0]
total = [4, 2, 1, 1]

x = np.arange(len(rule_types))
width = 0.25

bars1 = ax.bar(x - width, found_before, width, label='Before', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x, found_after, width, label='After', color='#27ae60', alpha=0.8)
bars3 = ax.bar(x + width, total, width, label='Total rules', color='#3498db', alpha=0.4)

ax.set_ylabel('Rules', fontsize=14)
ax.set_title('Rules Found by Property Type (Hard)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(rule_types)
ax.legend(loc='upper right')
ax.set_ylim(0, 5)

# Bottom right: Key findings
ax = axes[1, 1]
ax.axis('off')
findings = """
KEY FINDINGS

1. Scale action bug fixed
   - "Pyramids tip over on scale" now discoverable

2. Forced experiment diversity
   - Ensures all 8 experiment types get coverage
   - Prevents LLM bias toward "exciting" experiments

3. Lower evidence threshold (2 → 1)
   - Captures rare property+experiment combinations

4. Best configuration found:
   - 8 agents, 80 experiments each (640 total)
   - Communication: 1 peer every 2 experiments
   - Score: 87% (7/8 rules)

5. Remaining challenge:
   - Color-based rules hardest to discover
   - "Red objects are fireproof" often missed
"""
ax.text(0.1, 0.95, findings, transform=ax.transAxes, fontsize=12,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/summary_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved: figures/summary_dashboard.png")

print("\nAll graphs generated successfully!")
