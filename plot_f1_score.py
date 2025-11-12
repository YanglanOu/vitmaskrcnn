#!/usr/bin/env python3
"""
Script to plot F1 scores from training log file.
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_f1_scores(log_file):
    """Extract epoch numbers and F1 scores from log file."""
    epochs = []
    f1_scores = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to match "F1 Score: X.XXXX" in Validation Metrics section
    # We want to capture the first occurrence per epoch (in Validation Metrics)
    pattern = r'Validation Metrics:.*?F1 Score: ([\d.]+)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    # Convert to float and track epochs
    for idx, f1_str in enumerate(matches):
        epochs.append(idx)
        f1_scores.append(float(f1_str))
    
    return epochs, f1_scores

def plot_f1_scores(epochs, f1_scores, output_file=None):
    """Plot F1 scores vs epochs."""
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, f1_scores, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add some statistics
    max_f1 = max(f1_scores)
    max_epoch = epochs[f1_scores.index(max_f1)]
    plt.axhline(y=max_f1, color='r', linestyle='--', alpha=0.5, 
                label=f'Max F1: {max_f1:.4f} (epoch {max_epoch})')
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    log_file = Path('/home/m341664/yanglanou/projects/vitmaskrcnn/logs/20251111/vitg-14-fasterrcnn-fb-518_37MRSvitG_40007.out')
    
    print(f"Parsing F1 scores from {log_file}...")
    epochs, f1_scores = parse_f1_scores(log_file)
    
    print(f"Found {len(epochs)} epochs")
    print(f"F1 score range: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
    print(f"Best F1 score: {max(f1_scores):.4f} at epoch {epochs[f1_scores.index(max(f1_scores))]}")
    
    # Create output directory if it doesn't exist
    output_dir = Path('/home/m341664/yanglanou/projects/vitmaskrcnn/plots')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'f1_score_plot.png'
    plot_f1_scores(epochs, f1_scores, output_file)

