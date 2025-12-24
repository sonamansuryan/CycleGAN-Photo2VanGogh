import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import itertools
import json
from datetime import datetime
import glob

from models.networks import Generator, Discriminator, weights_init_normal
from utils.dataset import ImageDataset
from utils.helpers import ReplayBuffer, LambdaLR, save_sample_images, save_checkpoint


# ================== Configuration ==================
class Config:
    # Paths
    data_root = 'data/vangogh2photo'
    trainA_path = os.path.join(data_root, 'train', 'trainA')
    trainB_path = os.path.join(data_root, 'train', 'trainB')
    valA_path = os.path.join(data_root, 'val', 'testA')
    valB_path = os.path.join(data_root, 'val', 'testB')
    checkpoint_dir = 'checkpoints'
    sample_dir = 'samples'
    log_dir = 'logs'

    # Hyperparameters
    total_epochs = 300
    stop_at_epoch = None
    batch_size = 2
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    decay_epoch = 150

    # Loss weights
    lambda_cycle = 15.0
    lambda_identity = 0.1

    # Image
    img_size = 256
    img_channels = 3

    # Training
    sample_interval = 100
    checkpoint_interval = 5
    num_workers = 2

    # Validation & Overfitting Detection
    validate_every = 1  # Validate every N epochs
    early_stopping_patience = 20
    min_improvement = 0.001
    overfitting_threshold = 0.15  # If val_loss > train_loss + threshold, possible overfitting

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    """Early stopping based on validation loss"""

    def __init__(self, patience=20, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  No improvement for {self.counter}/{self.patience} epochs')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n[STOP] Early stopping triggered! Best was epoch {self.best_epoch}')
        else:
            if self.verbose:
                improvement = self.best_loss - val_loss
                print(f'  [IMPROVED] Loss improved by {improvement:.4f}')
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


def load_checkpoint(checkpoint_path, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B):
    """Load checkpoint and return the epoch number"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    has_optimizer_states = 'optimizer_G_state_dict' in checkpoint

    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
    D_A.load_state_dict(checkpoint['D_A_state_dict'])
    D_B.load_state_dict(checkpoint['D_B_state_dict'])

    if has_optimizer_states:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']} (with optimizer states)")
    else:
        print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']} (models only)")

    return checkpoint['epoch']


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None

    epochs = [int(cp.split('epoch_')[1].split('.pth')[0]) for cp in checkpoints]
    latest_idx = epochs.index(max(epochs))
    return checkpoints[latest_idx]


def get_augmented_transforms(img_size=256):
    """Reduced augmentation for better art style preservation"""
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.12), int(img_size * 1.12))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms(img_size=256):
    """No augmentation for validation"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def validate(G_AB, G_BA, D_A, D_B, val_loader, criterion_GAN, criterion_cycle,
             criterion_identity, config, device):
    """Validation loop to detect overfitting"""
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()

    val_loss_G = 0
    val_loss_D_A = 0
    val_loss_D_B = 0
    val_loss_cycle = 0
    val_loss_identity = 0

    with torch.no_grad():
        for batch in val_loader:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            batch_size = real_A.size(0)
            patch_size = 16
            valid = torch.ones((batch_size, 1, patch_size, patch_size)).to(device)
            fake = torch.zeros((batch_size, 1, patch_size, patch_size)).to(device)

            # Generator losses
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)

            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + config.lambda_cycle * loss_cycle + config.lambda_identity * loss_identity

            # Discriminator losses
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            loss_fake_A = criterion_GAN(D_A(fake_A), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

            loss_real_B = criterion_GAN(D_B(real_B), valid)
            loss_fake_B = criterion_GAN(D_B(fake_B), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

            val_loss_G += loss_G.item()
            val_loss_D_A += loss_D_A.item()
            val_loss_D_B += loss_D_B.item()
            val_loss_cycle += loss_cycle.item()
            val_loss_identity += loss_identity.item()

    # Set back to train mode
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    num_batches = len(val_loader)
    return {
        'G_loss': val_loss_G / num_batches,
        'D_A_loss': val_loss_D_A / num_batches,
        'D_B_loss': val_loss_D_B / num_batches,
        'cycle_loss': val_loss_cycle / num_batches,
        'identity_loss': val_loss_identity / num_batches,
        'total_loss': (val_loss_G + (val_loss_D_A + val_loss_D_B) / 2) / num_batches
    }


def train():
    config = Config()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f'training_log_{timestamp}.txt')
    loss_history_file = os.path.join(config.log_dir, f'loss_history_{timestamp}.json')

    def log_print(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    log_print(f"{'=' * 70}")
    log_print(f"TRAINING STARTED: {timestamp}")
    log_print(f"{'=' * 70}")
    log_print(f"Device: {config.device}")

    if config.device.type == 'cuda':
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    log_print(f"\nTraining Configuration:")
    log_print(f"  Total epochs: {config.total_epochs}")
    log_print(f"  Batch size: {config.batch_size}")
    log_print(f"  Lambda cycle: {config.lambda_cycle}")
    log_print(f"  Lambda identity: {config.lambda_identity}")
    log_print(f"  Decay starts at epoch: {config.decay_epoch}")
    log_print(f"  Validation every: {config.validate_every} epochs")
    log_print(f"  Overfitting threshold: {config.overfitting_threshold}")

    # ================== Data Loading ==================
    train_transform = get_augmented_transforms(config.img_size)
    val_transform = get_val_transforms(config.img_size)

    train_dataset = ImageDataset(config.trainA_path, config.trainB_path, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Check if validation data exists
    has_validation = os.path.exists(config.valA_path) and os.path.exists(config.valB_path)

    if has_validation:
        val_dataset = ImageDataset(config.valA_path, config.valB_path, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        log_print(f"\nDataset loaded:")
        log_print(f"  Training: {len(train_dataset)} image pairs")
        log_print(f"  Validation: {len(val_dataset)} image pairs")
        log_print(f"  Overfitting detection: ENABLED")
    else:
        log_print(f"\n[WARNING] Validation data not found in {config.valA_path} or {config.valB_path}")
        log_print(f"Training: {len(train_dataset)} image pairs")
        log_print(f"Overfitting detection: DISABLED (no validation data)")
        val_loader = None

    # ================== Models ==================
    G_AB = Generator(config.img_channels, num_residual_blocks=9).to(config.device)
    G_BA = Generator(config.img_channels, num_residual_blocks=9).to(config.device)
    D_A = Discriminator(config.img_channels).to(config.device)
    D_B = Discriminator(config.img_channels).to(config.device)

    # ================== Loss Functions ==================
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # ================== Optimizers ==================
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    # ================== Auto-Resume ==================
    start_epoch = 1
    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)

    if latest_checkpoint:
        try:
            start_epoch = load_checkpoint(latest_checkpoint, G_AB, G_BA, D_A, D_B,
                                          optimizer_G, optimizer_D_A, optimizer_D_B) + 1
            log_print(f"\n[RESUME] Continuing from epoch {start_epoch}")
        except Exception as e:
            log_print(f"\n[WARNING] Failed to load checkpoint: {e}")
            log_print("Starting fresh training...")
            G_AB.apply(weights_init_normal)
            G_BA.apply(weights_init_normal)
            D_A.apply(weights_init_normal)
            D_B.apply(weights_init_normal)
    else:
        log_print("\nNo checkpoint found. Starting fresh training...")
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    if config.stop_at_epoch is not None and start_epoch <= config.stop_at_epoch:
        target_epoch = config.stop_at_epoch
        log_print(f"Training from epoch {start_epoch} to {target_epoch}")
    else:
        target_epoch = config.total_epochs
        log_print(f"Training from epoch {start_epoch} to {target_epoch}")

    log_print(f"{'=' * 70}\n")

    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(config.total_epochs, start_epoch - 1, config.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=LambdaLR(config.total_epochs, start_epoch - 1, config.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=LambdaLR(config.total_epochs, start_epoch - 1, config.decay_epoch).step
    )

    # Replay buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.min_improvement,
        verbose=True
    )

    # Loss history
    loss_history = {
        'epoch': [],
        'train_G_loss': [],
        'train_D_A_loss': [],
        'train_D_B_loss': [],
        'train_cycle_loss': [],
        'train_identity_loss': [],
        'train_total_loss': [],
        'val_G_loss': [],
        'val_D_A_loss': [],
        'val_D_B_loss': [],
        'val_cycle_loss': [],
        'val_identity_loss': [],
        'val_total_loss': [],
        'overfitting_detected': []
    }

    best_val_loss = float('inf')
    overfitting_warnings = 0

    # ================== Training Loop ==================
    for epoch in range(start_epoch, target_epoch + 1):
        epoch_loss_G = 0
        epoch_loss_D_A = 0
        epoch_loss_D_B = 0
        epoch_loss_cycle = 0
        epoch_loss_identity = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{target_epoch}')

        for i, batch in enumerate(progress_bar):
            real_A = batch['A'].to(config.device)
            real_B = batch['B'].to(config.device)

            batch_size = real_A.size(0)

            patch_size = 16
            valid = torch.ones((batch_size, 1, patch_size, patch_size), requires_grad=False).to(config.device)
            fake = torch.zeros((batch_size, 1, patch_size, patch_size), requires_grad=False).to(config.device)

            # ================== Train Generators ==================
            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)

            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + config.lambda_cycle * loss_cycle + config.lambda_identity * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminators ==================
            optimizer_D_A.zero_grad()

            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_buffered = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_buffered.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_buffered = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_buffered.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()
            epoch_loss_cycle += loss_cycle.item()
            epoch_loss_identity += loss_identity.item()

            progress_bar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D': f'{(loss_D_A.item() + loss_D_B.item()) / 2:.4f}',
                'Cyc': f'{loss_cycle.item():.4f}'
            })

            if i % config.sample_interval == 0:
                with torch.no_grad():
                    save_sample_images(real_A, real_B, fake_A, fake_B, epoch, config.sample_dir)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Calculate training averages
        avg_train_loss_G = epoch_loss_G / len(train_loader)
        avg_train_loss_D_A = epoch_loss_D_A / len(train_loader)
        avg_train_loss_D_B = epoch_loss_D_B / len(train_loader)
        avg_train_loss_cycle = epoch_loss_cycle / len(train_loader)
        avg_train_loss_identity = epoch_loss_identity / len(train_loader)
        train_total_loss = avg_train_loss_G + (avg_train_loss_D_A + avg_train_loss_D_B) / 2

        # Validation
        overfitting_detected = False
        if has_validation and epoch % config.validate_every == 0:
            val_losses = validate(G_AB, G_BA, D_A, D_B, val_loader, criterion_GAN,
                                  criterion_cycle, criterion_identity, config, config.device)
            val_total_loss = val_losses['total_loss']

            # Detect overfitting
            loss_gap = val_total_loss - train_total_loss
            if loss_gap > config.overfitting_threshold:
                overfitting_detected = True
                overfitting_warnings += 1
        else:
            val_losses = None
            val_total_loss = None

        # Save to history
        loss_history['epoch'].append(epoch)
        loss_history['train_G_loss'].append(avg_train_loss_G)
        loss_history['train_D_A_loss'].append(avg_train_loss_D_A)
        loss_history['train_D_B_loss'].append(avg_train_loss_D_B)
        loss_history['train_cycle_loss'].append(avg_train_loss_cycle)
        loss_history['train_identity_loss'].append(avg_train_loss_identity)
        loss_history['train_total_loss'].append(train_total_loss)

        if val_losses:
            loss_history['val_G_loss'].append(val_losses['G_loss'])
            loss_history['val_D_A_loss'].append(val_losses['D_A_loss'])
            loss_history['val_D_B_loss'].append(val_losses['D_B_loss'])
            loss_history['val_cycle_loss'].append(val_losses['cycle_loss'])
            loss_history['val_identity_loss'].append(val_losses['identity_loss'])
            loss_history['val_total_loss'].append(val_total_loss)
            loss_history['overfitting_detected'].append(overfitting_detected)
        else:
            loss_history['val_G_loss'].append(None)
            loss_history['val_D_A_loss'].append(None)
            loss_history['val_D_B_loss'].append(None)
            loss_history['val_cycle_loss'].append(None)
            loss_history['val_identity_loss'].append(None)
            loss_history['val_total_loss'].append(None)
            loss_history['overfitting_detected'].append(False)

        # Print summary
        summary = f"\nEpoch {epoch}/{target_epoch} Summary:"
        summary += f"\n  [TRAIN] G Loss: {avg_train_loss_G:.4f}"
        summary += f"\n  [TRAIN] D_A Loss: {avg_train_loss_D_A:.4f}"
        summary += f"\n  [TRAIN] D_B Loss: {avg_train_loss_D_B:.4f}"
        summary += f"\n  [TRAIN] Cycle Loss: {avg_train_loss_cycle:.4f}"
        summary += f"\n  [TRAIN] Identity Loss: {avg_train_loss_identity:.4f}"
        summary += f"\n  [TRAIN] Total Loss: {train_total_loss:.4f}"

        if val_losses:
            summary += f"\n  [VAL] Total Loss: {val_total_loss:.4f}"
            summary += f"\n  [VAL] Loss Gap: {loss_gap:.4f}"
            if overfitting_detected:
                summary += f"\n  [WARNING] Possible overfitting detected! (Gap > {config.overfitting_threshold})"
                summary += f"\n  [WARNING] Total overfitting warnings: {overfitting_warnings}"

        summary += f"\n  Learning Rate: {optimizer_G.param_groups[0]['lr']:.6f}"

        # Save best model based on validation or training loss
        comparison_loss = val_total_loss if val_losses else train_total_loss
        if comparison_loss < best_val_loss:
            best_val_loss = comparison_loss
            summary += f"\n  [BEST] New best model saved!"
            torch.save(G_AB.state_dict(), f'{config.checkpoint_dir}/G_AB_best.pth')
            torch.save(G_BA.state_dict(), f'{config.checkpoint_dir}/G_BA_best.pth')

        log_print(summary)

        with open(loss_history_file, 'w') as f:
            json.dump(loss_history, f, indent=2)

        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(
                epoch, G_AB, G_BA, D_A, D_B,
                optimizer_G, optimizer_D_A, optimizer_D_B,
                config.checkpoint_dir
            )

        # Early stopping based on validation loss if available
        if val_losses:
            if early_stopping(epoch, val_total_loss):
                log_print(f"\n{'=' * 70}")
                log_print(f"[STOP] Early stopping triggered at epoch {epoch}")
                log_print(f"[STOP] Best epoch was: {early_stopping.best_epoch}")
                log_print(f"{'=' * 70}")
                break

        if epoch == target_epoch and target_epoch < config.total_epochs:
            log_print(f"\n{'=' * 70}")
            log_print(f"[CHECKPOINT] Reached stopping point at epoch {epoch}")
            log_print(f"Run again to continue from epoch {epoch + 1}")
            log_print(f"{'=' * 70}")
            save_checkpoint(
                epoch, G_AB, G_BA, D_A, D_B,
                optimizer_G, optimizer_D_A, optimizer_D_B,
                config.checkpoint_dir
            )
            break

    # Save final model
    if epoch >= config.total_epochs:
        torch.save(G_AB.state_dict(), f'{config.checkpoint_dir}/G_AB_final.pth')
        torch.save(G_BA.state_dict(), f'{config.checkpoint_dir}/G_BA_final.pth')
        log_print(f"\n{'=' * 70}")
        log_print(f"[DONE] Training completed! All {config.total_epochs} epochs finished.")
        if overfitting_warnings > 0:
            log_print(f"[INFO] Total overfitting warnings during training: {overfitting_warnings}")
        log_print(f"{'=' * 70}")


if __name__ == '__main__':
    train()