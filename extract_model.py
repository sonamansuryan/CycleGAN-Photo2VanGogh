import torch
from models.networks import Generator

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_260.pth', weights_only=False)

# Extract and save G_AB
G_AB = Generator(num_residual_blocks=9)
G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
torch.save(G_AB.state_dict(), 'checkpoints/G_AB_epoch260.pth')

print("[OK] Saved G_AB_epoch260.pth")

# Extract and save G_BA
G_BA = Generator(num_residual_blocks=9)
G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
torch.save(G_BA.state_dict(), 'checkpoints/G_BA_epoch260.pth')

print("[OK] Saved G_BA_epoch260.pth")