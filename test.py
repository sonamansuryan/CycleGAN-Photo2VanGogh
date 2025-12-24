import torch
from torchvision import transforms
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import argparse

from models.networks import Generator


def load_image(image_path, img_size=256):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def denormalize(tensor):
    """Convert normalized tensor to displayable image"""
    tensor = (tensor + 1) / 2
    return torch.clamp(tensor, 0, 1)


def test_single_image(model_path, image_path, output_path=None, device='cuda', show=False):
    """
    Test model on a single image

    Args:
        model_path: Path to saved generator model (.pth file)
        image_path: Path to input image
        output_path: Path to save result (optional)
        device: 'cuda' or 'cpu'
        show: Whether to display the result (set False for server environments)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    G = Generator(num_residual_blocks=9).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    G.eval()
    print(f"[OK] Model loaded from {model_path}")

    # Load and process image
    input_img = load_image(image_path).to(device)
    print(f"[OK] Image loaded: {image_path}")

    # Generate output
    with torch.no_grad():
        output_img = G(input_img)

    # Convert to displayable format
    input_display = denormalize(input_img.squeeze(0)).cpu()
    output_display = denormalize(output_img.squeeze(0)).cpu()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(input_display.permute(1, 2, 0))
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(output_display.permute(1, 2, 0))
    axes[1].set_title('Van Gogh Style', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Comparison saved to {output_path}")

        # Also save just the output image
        output_img_path = output_path.replace('.png', '_output.png')
        transforms.ToPILImage()(output_display).save(output_img_path)
        print(f"[OK] Output image saved to {output_img_path}")

    if show:
        plt.show()

    plt.close(fig)
    print("[OK] Processing complete!\n")


def test_directory(model_path, input_dir, output_dir, device='cuda', save_comparison=True):
    """
    Test model on all images in a directory

    Args:
        model_path: Path to saved generator model
        input_dir: Directory containing input images
        output_dir: Directory to save results
        device: 'cuda' or 'cpu'
        save_comparison: Whether to save side-by-side comparison images
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if save_comparison:
        comparison_dir = os.path.join(output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)

    # Load model
    G = Generator(num_residual_blocks=9).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    G.eval()
    print(f"[OK] Model loaded from {model_path}\n")

    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process\n")
    print("=" * 70)

    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")

        input_path = os.path.join(input_dir, image_file)
        output_filename = f"vangogh_{os.path.splitext(image_file)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Load image
            input_img = load_image(input_path).to(device)

            # Generate output
            with torch.no_grad():
                output_img = G(input_img)

            # Save output image
            output_display = denormalize(output_img.squeeze(0)).cpu()
            transforms.ToPILImage()(output_display).save(output_path)
            print(f"  [OK] Saved to: {output_path}")

            # Save comparison if requested
            if save_comparison:
                input_display = denormalize(input_img.squeeze(0)).cpu()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                axes[0].imshow(input_display.permute(1, 2, 0))
                axes[0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                axes[1].imshow(output_display.permute(1, 2, 0))
                axes[1].set_title('Van Gogh Style', fontsize=12, fontweight='bold')
                axes[1].axis('off')

                plt.tight_layout()

                comparison_path = os.path.join(comparison_dir, f"comparison_{output_filename}")
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  [OK] Comparison saved to: {comparison_path}")

        except Exception as e:
            print(f"  [ERROR] Error processing {image_file}: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"[DONE] Processing complete! Results saved to: {output_dir}")
    if save_comparison:
        print(f"[DONE] Comparisons saved to: {comparison_dir}")


def test_both_directions(model_AB_path, model_BA_path, image_path, output_path, device='cuda'):
    """
    Test both directions: Photo→VanGogh and VanGogh→Photo

    Args:
        model_AB_path: Path to G_AB (Photo to VanGogh) model
        model_BA_path: Path to G_BA (VanGogh to Photo) model
        image_path: Path to input image
        output_path: Path to save result
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load both models
    G_AB = Generator(num_residual_blocks=9).to(device)
    G_BA = Generator(num_residual_blocks=9).to(device)

    G_AB.load_state_dict(torch.load(model_AB_path, map_location=device, weights_only=False))
    G_BA.load_state_dict(torch.load(model_BA_path, map_location=device, weights_only=False))

    G_AB.eval()
    G_BA.eval()
    print(f"[OK] Models loaded")

    # Load image
    input_img = load_image(image_path).to(device)

    # Generate outputs
    with torch.no_grad():
        output_AB = G_AB(input_img)  # Photo → VanGogh
        output_BA = G_BA(input_img)  # VanGogh → Photo
        reconstructed = G_BA(output_AB)  # Photo → VanGogh → Photo (cycle)

    # Convert to displayable format
    input_display = denormalize(input_img.squeeze(0)).cpu()
    output_AB_display = denormalize(output_AB.squeeze(0)).cpu()
    output_BA_display = denormalize(output_BA.squeeze(0)).cpu()
    reconstructed_display = denormalize(reconstructed.squeeze(0)).cpu()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(input_display.permute(1, 2, 0))
    axes[0, 0].set_title('Original Input', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(output_AB_display.permute(1, 2, 0))
    axes[0, 1].set_title('Photo → Van Gogh', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(output_BA_display.permute(1, 2, 0))
    axes[1, 0].set_title('Van Gogh → Photo', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(reconstructed_display.permute(1, 2, 0))
    axes[1, 1].set_title('Cycle Reconstructed', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CycleGAN model')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'directory', 'both'],
                        help='Test mode: single image, directory, or both directions')
    parser.add_argument('--model', type=str, default='checkpoints/G_AB_best.pth',
                        help='Path to generator model')
    parser.add_argument('--model_ba', type=str, default='checkpoints/G_BA_best.pth',
                        help='Path to reverse generator (for both mode)')
    parser.add_argument('--input', type=str, default='test_images/photo.jpg',
                        help='Input image or directory path')
    parser.add_argument('--output', type=str, default='results',
                        help='Output path or directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--no-comparison', action='store_true',
                        help='Skip saving comparison images in directory mode')

    args = parser.parse_args()

    # Ensure output directory exists
    if args.mode == 'single':
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'results', exist_ok=True)
    else:
        os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("CycleGAN Image Style Transfer - Testing")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 70 + "\n")

    if args.mode == 'single':
        output_file = args.output if args.output.endswith('.png') else os.path.join(args.output, 'result.png')
        test_single_image(
            model_path=args.model,
            image_path=args.input,
            output_path=output_file,
            device=args.device,
            show=False
        )

    elif args.mode == 'directory':
        test_directory(
            model_path=args.model,
            input_dir=args.input,
            output_dir=args.output,
            device=args.device,
            save_comparison=not args.no_comparison
        )

    elif args.mode == 'both':
        output_file = args.output if args.output.endswith('.png') else os.path.join(args.output, 'both_directions.png')
        test_both_directions(
            model_AB_path=args.model,
            model_BA_path=args.model_ba,
            image_path=args.input,
            output_path=output_file,
            device=args.device
        )