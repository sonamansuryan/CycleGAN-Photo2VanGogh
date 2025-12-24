import glob
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    """Dataset for unpaired image-to-image translation"""

    def __init__(self, root_A, root_B, transform=None):
        """
        Args:
            root_A: Path to domain A images (e.g., Van Gogh paintings)
            root_B: Path to domain B images (e.g., real photos)
            transform: Torchvision transforms to apply
        """
        self.transform = transform

        # Get all image files
        self.files_A = sorted(glob.glob(os.path.join(root_A, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root_B, '*.*')))

        # Use maximum length for unpaired data
        self.length = max(len(self.files_A), len(self.files_B))

        print(f"Dataset initialized:")
        print(f"  Domain A: {len(self.files_A)} images from {root_A}")
        print(f"  Domain B: {len(self.files_B)} images from {root_B}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Use modulo to handle different dataset sizes
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert('RGB')
        img_B = Image.open(self.files_B[idx % len(self.files_B)]).convert('RGB')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}