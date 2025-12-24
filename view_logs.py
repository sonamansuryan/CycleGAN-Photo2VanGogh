import os
import json
import matplotlib.pyplot as plt
import glob


def view_latest_log():
    """View the latest training log"""
    log_dir = 'logs'

    if not os.path.exists(log_dir):
        print("No logs directory found. Run training first.")
        return

    # Find latest log file
    log_files = glob.glob(os.path.join(log_dir, 'training_log_*.txt'))

    if not log_files:
        print("No log files found.")
        return

    latest_log = max(log_files, key=os.path.getctime)

    print(f"Reading log file: {latest_log}\n")
    print("=" * 60)

    with open(latest_log, 'r') as f:
        content = f.read()
        print(content)

    print("=" * 60)


def view_loss_history():
    """View and plot loss history"""
    log_dir = 'logs'

    if not os.path.exists(log_dir):
        print("No logs directory found. Run training first.")
        return

    # Find latest loss history file
    loss_files = glob.glob(os.path.join(log_dir, 'loss_history_*.json'))

    if not loss_files:
        print("No loss history files found.")
        return

    latest_loss_file = max(loss_files, key=os.path.getctime)

    print(f"\nReading loss history: {latest_loss_file}\n")

    with open(latest_loss_file, 'r') as f:
        loss_history = json.load(f)

    # Print last 10 epochs
    print("Last 10 epochs:")
    print("-" * 80)
    print(f"{'Epoch':<8} {'G Loss':<12} {'D_A Loss':<12} {'D_B Loss':<12} {'Cycle':<12} {'Identity':<12}")
    print("-" * 80)

    for i in range(max(0, len(loss_history['epoch']) - 10), len(loss_history['epoch'])):
        epoch = loss_history['epoch'][i]
        g_loss = loss_history['G_loss'][i]
        d_a_loss = loss_history['D_A_loss'][i]
        d_b_loss = loss_history['D_B_loss'][i]
        cycle_loss = loss_history['cycle_loss'][i]
        identity_loss = loss_history['identity_loss'][i]

        print(
            f"{epoch:<8} {g_loss:<12.4f} {d_a_loss:<12.4f} {d_b_loss:<12.4f} {cycle_loss:<12.4f} {identity_loss:<12.4f}")

    print("-" * 80)

    # Plot losses
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generator and Discriminator losses
    axes[0, 0].plot(loss_history['epoch'], loss_history['G_loss'], label='Generator', color='blue')
    axes[0, 0].plot(loss_history['epoch'], loss_history['D_A_loss'], label='Discriminator A', color='red', alpha=0.7)
    axes[0, 0].plot(loss_history['epoch'], loss_history['D_B_loss'], label='Discriminator B', color='orange', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator and Discriminator Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cycle consistency loss
    axes[0, 1].plot(loss_history['epoch'], loss_history['cycle_loss'], color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Cycle Consistency Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Identity loss
    axes[1, 0].plot(loss_history['epoch'], loss_history['identity_loss'], color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Identity Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # All losses combined
    axes[1, 1].plot(loss_history['epoch'], loss_history['G_loss'], label='Generator', linewidth=2)
    axes[1, 1].plot(loss_history['epoch'], loss_history['cycle_loss'], label='Cycle', alpha=0.7)
    axes[1, 1].plot(loss_history['epoch'], loss_history['identity_loss'], label='Identity', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(log_dir, 'loss_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss plot saved to: {plot_path}")

    plt.show()


def view_checkpoints():
    """List all saved checkpoints"""
    checkpoint_dir = 'checkpoints'

    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found.")
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    checkpoints.sort()

    if not checkpoints:
        print("No checkpoints found.")
        return

    print("\nAvailable checkpoints:")
    print("-" * 60)
    for cp in checkpoints:
        filename = os.path.basename(cp)
        size_mb = os.path.getsize(cp) / (1024 * 1024)
        print(f"  {filename:<30} ({size_mb:.1f} MB)")

    # Check for final models
    if os.path.exists(os.path.join(checkpoint_dir, 'G_AB_final.pth')):
        size_mb = os.path.getsize(os.path.join(checkpoint_dir, 'G_AB_final.pth')) / (1024 * 1024)
        print(f"  {'G_AB_final.pth':<30} ({size_mb:.1f} MB) ✓ FINAL")

    if os.path.exists(os.path.join(checkpoint_dir, 'G_BA_final.pth')):
        size_mb = os.path.getsize(os.path.join(checkpoint_dir, 'G_BA_final.pth')) / (1024 * 1024)
        print(f"  {'G_BA_final.pth':<30} ({size_mb:.1f} MB) ✓ FINAL")

    print("-" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TRAINING RESULTS VIEWER")
    print("=" * 60)

    # View checkpoints
    view_checkpoints()

    # View latest log
    view_latest_log()

    # View and plot loss history
    view_loss_history()