
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from glob import glob

# Configuration
LOGS_DIR = 'logs'
OUTPUT_DIR = 'analysis_results'
EPOCH_START = 150
EPOCH_END = 260


def find_latest_log():
    """Find the most recent training log file"""
    log_files = glob(os.path.join(LOGS_DIR, 'loss_history_*.json'))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {LOGS_DIR}")
    return max(log_files, key=os.path.getmtime)


def load_training_data(log_file):
    """Load training data from JSON log file"""
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data


def filter_epoch_range(data, start_epoch, end_epoch=None):
    """Filter data for specific epoch range"""
    epochs = np.array(data['epoch'])
    if end_epoch is None:
        # If end_epoch is None, go to the end of available data
        mask = epochs >= start_epoch
        actual_end = int(epochs[-1]) if len(epochs) > 0 else start_epoch
    else:
        mask = (epochs >= start_epoch) & (epochs <= end_epoch)
        actual_end = end_epoch

    filtered_data = {}
    for key, values in data.items():
        if isinstance(values, list):
            filtered_data[key] = [v for i, v in enumerate(values) if mask[i]]
        else:
            filtered_data[key] = values

    return filtered_data, actual_end


def calculate_statistics(data, start_epoch, end_epoch=None):
    """Calculate statistics for the epoch range"""
    filtered, actual_end = filter_epoch_range(data, start_epoch, end_epoch)

    stats = {
        'epoch_range': f"{start_epoch}-{actual_end}",
        'total_epochs': len(filtered['epoch']),
    }

    # Calculate statistics for each metric
    metrics = ['train_G_loss', 'train_D_A_loss', 'train_D_B_loss',
               'train_cycle_loss', 'train_identity_loss', 'train_total_loss']

    if filtered.get('val_total_loss') and any(v is not None for v in filtered['val_total_loss']):
        metrics.extend(['val_G_loss', 'val_D_A_loss', 'val_D_B_loss',
                       'val_cycle_loss', 'val_identity_loss', 'val_total_loss'])

    for metric in metrics:
        if metric in filtered and filtered[metric]:
            values = [v for v in filtered[metric] if v is not None]
            if values:
                stats[metric] = {
                    'start': values[0],
                    'end': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'change': values[-1] - values[0],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }

    # Overfitting detection
    if 'overfitting_detected' in filtered:
        overfitting_count = sum(filtered['overfitting_detected'])
        stats['overfitting_warnings'] = overfitting_count
        if len(filtered['epoch']) > 0:
            stats['overfitting_percentage'] = (overfitting_count / len(filtered['epoch']) * 100)
        else:
            stats['overfitting_percentage'] = 0.0

    return stats


def plot_training_progression(data, start_epoch, end_epoch, output_dir):
    """Create comprehensive visualization of training progression"""
    filtered, actual_end = filter_epoch_range(data, start_epoch, end_epoch)
    epochs = filtered['epoch']

    os.makedirs(output_dir, exist_ok=True)

    # Determine if we have validation data
    has_validation = (filtered.get('val_total_loss') and
                     any(v is not None for v in filtered['val_total_loss']))

    # Create figure with subplots
    if has_validation:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Training Progression Analysis: Epochs {start_epoch}-{actual_end}',
                    fontsize=16, fontweight='bold', y=0.995)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Training Progression Analysis: Epochs {start_epoch}-{actual_end}',
                    fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten()

    # 1. Generator Loss
    ax = axes[0]
    ax.plot(epochs, filtered['train_G_loss'], 'b-', linewidth=2, label='Train G Loss', alpha=0.8)
    if has_validation:
        val_g = [v for v in filtered['val_G_loss'] if v is not None]
        val_epochs = [e for e, v in zip(epochs, filtered['val_G_loss']) if v is not None]
        ax.plot(val_epochs, val_g, 'r--', linewidth=2, label='Val G Loss', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Generator Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Discriminator Losses
    ax = axes[1]
    ax.plot(epochs, filtered['train_D_A_loss'], 'g-', linewidth=2, label='Train D_A', alpha=0.8)
    ax.plot(epochs, filtered['train_D_B_loss'], 'm-', linewidth=2, label='Train D_B', alpha=0.8)
    if has_validation:
        val_da = [v for v in filtered['val_D_A_loss'] if v is not None]
        val_db = [v for v in filtered['val_D_B_loss'] if v is not None]
        val_epochs = [e for e, v in zip(epochs, filtered['val_D_A_loss']) if v is not None]
        ax.plot(val_epochs, val_da, 'g--', linewidth=2, label='Val D_A', alpha=0.8)
        ax.plot(val_epochs, val_db, 'm--', linewidth=2, label='Val D_B', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Discriminator Losses', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Cycle Loss
    ax = axes[2]
    ax.plot(epochs, filtered['train_cycle_loss'], 'c-', linewidth=2, label='Train Cycle', alpha=0.8)
    if has_validation:
        val_cycle = [v for v in filtered['val_cycle_loss'] if v is not None]
        val_epochs = [e for e, v in zip(epochs, filtered['val_cycle_loss']) if v is not None]
        ax.plot(val_epochs, val_cycle, 'c--', linewidth=2, label='Val Cycle', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Cycle Consistency Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 4. Identity Loss
    ax = axes[3]
    ax.plot(epochs, filtered['train_identity_loss'], 'y-', linewidth=2, label='Train Identity', alpha=0.8)
    if has_validation:
        val_id = [v for v in filtered['val_identity_loss'] if v is not None]
        val_epochs = [e for e, v in zip(epochs, filtered['val_identity_loss']) if v is not None]
        ax.plot(val_epochs, val_id, 'y--', linewidth=2, label='Val Identity', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Identity Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 5. Total Loss Comparison
    ax = axes[4]
    ax.plot(epochs, filtered['train_total_loss'], 'b-', linewidth=2.5, label='Train Total', alpha=0.8)
    if has_validation:
        val_total = [v for v in filtered['val_total_loss'] if v is not None]
        val_epochs = [e for e, v in zip(epochs, filtered['val_total_loss']) if v is not None]
        ax.plot(val_epochs, val_total, 'r-', linewidth=2.5, label='Val Total', alpha=0.8)

        # Highlight overfitting regions
        overfitting_epochs = [e for e, o in zip(epochs, filtered['overfitting_detected']) if o]
        if overfitting_epochs:
            for epoch in overfitting_epochs:
                ax.axvline(x=epoch, color='orange', alpha=0.3, linewidth=1)

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Total Loss (Train vs Validation)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 6. Loss Change Rate
    ax = axes[5]
    train_total = filtered['train_total_loss']
    train_change = np.diff(train_total)
    ax.plot(epochs[1:], train_change, 'b-', linewidth=2, label='Train Loss Change', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    if has_validation:
        val_total = [v for v in filtered['val_total_loss'] if v is not None]
        if len(val_total) > 1:
            val_change = np.diff(val_total)
            val_epochs_diff = val_epochs[1:]
            ax.plot(val_epochs_diff, val_change, 'r-', linewidth=2, label='Val Loss Change', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss Change', fontsize=11, fontweight='bold')
    ax.set_title('Loss Change Rate (Epoch-to-Epoch)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'training_progression_{start_epoch}_{actual_end}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Main visualization saved to: {output_path}")
    return output_path


def plot_loss_components_breakdown(data, start_epoch, end_epoch, output_dir):
    """Create detailed breakdown of loss components"""
    filtered, actual_end = filter_epoch_range(data, start_epoch, end_epoch)
    epochs = filtered['epoch']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Loss Components Breakdown: Epochs {start_epoch}-{actual_end}',
                fontsize=16, fontweight='bold')

    # Training losses stacked
    ax = axes[0]
    ax.plot(epochs, filtered['train_G_loss'], linewidth=2, label='Generator', marker='o', markersize=3)
    ax.plot(epochs, filtered['train_D_A_loss'], linewidth=2, label='Discriminator A', marker='s', markersize=3)
    ax.plot(epochs, filtered['train_D_B_loss'], linewidth=2, label='Discriminator B', marker='^', markersize=3)
    ax.plot(epochs, filtered['train_cycle_loss'], linewidth=2, label='Cycle Consistency', marker='d', markersize=3)
    ax.plot(epochs, filtered['train_identity_loss'], linewidth=2, label='Identity', marker='*', markersize=4)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax.set_title('Training Loss Components', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Loss ratios
    ax = axes[1]
    total = np.array(filtered['train_total_loss'])
    g_ratio = np.array(filtered['train_G_loss']) / total * 100
    d_ratio = (np.array(filtered['train_D_A_loss']) + np.array(filtered['train_D_B_loss'])) / 2 / total * 100
    cycle_ratio = np.array(filtered['train_cycle_loss']) / total * 100

    ax.plot(epochs, g_ratio, linewidth=2, label='Generator %', marker='o', markersize=3)
    ax.plot(epochs, d_ratio, linewidth=2, label='Discriminator %', marker='s', markersize=3)
    ax.plot(epochs, cycle_ratio, linewidth=2, label='Cycle %', marker='d', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Total Loss', fontsize=11, fontweight='bold')
    ax.set_title('Loss Component Ratios', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'loss_breakdown_{start_epoch}_{actual_end}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Loss breakdown visualization saved to: {output_path}")
    return output_path


def generate_report(stats, output_dir):
    """Generate text report of statistics"""
    report_path = os.path.join(output_dir, 'training_analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TRAINING PROGRESSION ANALYSIS: EPOCHS {stats['epoch_range']}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Epochs Analyzed: {stats['total_epochs']}\n\n")

        # Training metrics
        f.write("-" * 80 + "\n")
        f.write("TRAINING METRICS\n")
        f.write("-" * 80 + "\n\n")

        metrics = [
            ('Generator Loss', 'train_G_loss'),
            ('Discriminator A Loss', 'train_D_A_loss'),
            ('Discriminator B Loss', 'train_D_B_loss'),
            ('Cycle Loss', 'train_cycle_loss'),
            ('Identity Loss', 'train_identity_loss'),
            ('Total Loss', 'train_total_loss')
        ]

        for name, key in metrics:
            if key in stats:
                metric_data = stats[key]
                f.write(f"{name}:\n")
                f.write(f"  Start (Epoch {stats['epoch_range'].split('-')[0]}): {metric_data['start']:.6f}\n")
                f.write(f"  End (Epoch {stats['epoch_range'].split('-')[1]}): {metric_data['end']:.6f}\n")
                f.write(f"  Change: {metric_data['change']:.6f} ({metric_data['change_percent']:+.2f}%)\n")
                f.write(f"  Mean: {metric_data['mean']:.6f}\n")
                f.write(f"  Std Dev: {metric_data['std']:.6f}\n")
                f.write(f"  Min: {metric_data['min']:.6f}\n")
                f.write(f"  Max: {metric_data['max']:.6f}\n")
                f.write("\n")

        # Validation metrics
        if 'val_total_loss' in stats:
            f.write("-" * 80 + "\n")
            f.write("VALIDATION METRICS\n")
            f.write("-" * 80 + "\n\n")

            val_metrics = [
                ('Generator Loss', 'val_G_loss'),
                ('Discriminator A Loss', 'val_D_A_loss'),
                ('Discriminator B Loss', 'val_D_B_loss'),
                ('Cycle Loss', 'val_cycle_loss'),
                ('Identity Loss', 'val_identity_loss'),
                ('Total Loss', 'val_total_loss')
            ]

            for name, key in val_metrics:
                if key in stats:
                    metric_data = stats[key]
                    f.write(f"{name}:\n")
                    f.write(f"  Start: {metric_data['start']:.6f}\n")
                    f.write(f"  End: {metric_data['end']:.6f}\n")
                    f.write(f"  Change: {metric_data['change']:.6f} ({metric_data['change_percent']:+.2f}%)\n")
                    f.write(f"  Mean: {metric_data['mean']:.6f}\n")
                    f.write(f"  Std Dev: {metric_data['std']:.6f}\n")
                    f.write("\n")

        # Overfitting analysis
        if 'overfitting_warnings' in stats:
            f.write("-" * 80 + "\n")
            f.write("OVERFITTING ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Overfitting Warnings: {stats['overfitting_warnings']}\n")
            f.write(f"Percentage of Epochs: {stats['overfitting_percentage']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Text report saved to: {report_path}")
    return report_path


def main():
    """Main analysis function"""
    print("=" * 80)
    print("CycleGAN Training Progression Analysis")
    if EPOCH_END is None:
        print(f"Analyzing epochs {EPOCH_START} to end of training")
    else:
        print(f"Analyzing epochs {EPOCH_START} to {EPOCH_END}")
    print("=" * 80)
    print()

    # Find and load log file
    try:
        log_file = find_latest_log()
        print(f"[OK] Found log file: {log_file}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Load data
    print("[OK] Loading training data...")
    data = load_training_data(log_file)
    print(f"[OK] Loaded data for {len(data['epoch'])} epochs")

    # Calculate statistics
    if EPOCH_END is None:
        print(f"[OK] Calculating statistics for epochs {EPOCH_START} to end...")
    else:
        print(f"[OK] Calculating statistics for epochs {EPOCH_START}-{EPOCH_END}...")
    stats = calculate_statistics(data, EPOCH_START, EPOCH_END)

    # Create visualizations
    print("[OK] Creating visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main_plot = plot_training_progression(data, EPOCH_START, EPOCH_END, OUTPUT_DIR)
    breakdown_plot = plot_loss_components_breakdown(data, EPOCH_START, EPOCH_END, OUTPUT_DIR)

    # Generate report
    print("[OK] Generating text report...")
    report = generate_report(stats, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print(f"  - Main visualization: {os.path.basename(main_plot)}")
    print(f"  - Loss breakdown: {os.path.basename(breakdown_plot)}")
    print(f"  - Text report: {os.path.basename(report)}")
    print()

    # Print key findings
    print("KEY FINDINGS:")
    print("-" * 80)
    if 'train_total_loss' in stats:
        train_change = stats['train_total_loss']['change_percent']
        print(f"Training Total Loss Change: {train_change:+.2f}%")

    if 'val_total_loss' in stats:
        val_change = stats['val_total_loss']['change_percent']
        print(f"Validation Total Loss Change: {val_change:+.2f}%")

    if 'overfitting_warnings' in stats:
        print(f"Overfitting Warnings: {stats['overfitting_warnings']} ({stats['overfitting_percentage']:.1f}% of epochs)")

    print("=" * 80)


if __name__ == '__main__':
    main()