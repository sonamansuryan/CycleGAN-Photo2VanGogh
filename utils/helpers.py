import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


class ReplayBuffer:
    """Replay buffer to store previously generated images"""

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        Return images from buffer and update buffer with new images
        With 50% probability, return stored image instead of current
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR:
    """Learning rate scheduler with linear decay"""

    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epochs - self.decay_epoch)


def save_sample_images(real_A, real_B, fake_A, fake_B, epoch, save_path):
    """Save sample images during training (thread-safe version)"""

    def denormalize(tensor):
        return (tensor + 1) / 2

    # Create grid
    real_A = denormalize(real_A[:4])
    real_B = denormalize(real_B[:4])
    fake_A = denormalize(fake_A[:4])
    fake_B = denormalize(fake_B[:4])

    # Arrange images
    img_grid = torch.cat((real_A, fake_B, real_B, fake_A), dim=0)
    grid = make_grid(img_grid, nrow=4, normalize=False)

    # Plot with explicit figure creation and closure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    ax.set_title(f'Epoch {epoch}\nTop: Real Photo → Fake VanGogh | Bottom: Real VanGogh → Fake Photo')

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save and explicitly close
    save_file = os.path.join(save_path, f'epoch_{epoch}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close figure to prevent memory leaks
    plt.close('all')  # Close all figures to be extra safe


def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
    }

    os.makedirs(save_path, exist_ok=True)
    checkpoint_file = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_file)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(checkpoint_path, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
    D_A.load_state_dict(checkpoint['D_A_state_dict'])
    D_B.load_state_dict(checkpoint['D_B_state_dict'])

    # Handle both old and new checkpoint formats
    optimizer_key_G = 'optimizer_G_state_dict' if 'optimizer_G_state_dict' in checkpoint else 'optimizer_G'
    optimizer_key_D_A = 'optimizer_D_A_state_dict' if 'optimizer_D_A_state_dict' in checkpoint else 'optimizer_D_A'
    optimizer_key_D_B = 'optimizer_D_B_state_dict' if 'optimizer_D_B_state_dict' in checkpoint else 'optimizer_D_B'

    optimizer_G.load_state_dict(checkpoint[optimizer_key_G])
    optimizer_D_A.load_state_dict(checkpoint[optimizer_key_D_A])
    optimizer_D_B.load_state_dict(checkpoint[optimizer_key_D_B])

    epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from epoch {epoch}')
    return epoch