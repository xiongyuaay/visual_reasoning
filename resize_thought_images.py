#!/usr/bin/env python3
"""
Resize thought images to reduce sequence length and memory usage.
Keeps aspect ratio and ensures text remains readable.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

from PIL import Image
from tqdm import tqdm


def resize_image(
    img: Image.Image,
    target_width: int = 450,
    target_height: int = 600,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio within target dimensions.
    Uses high-quality LANCZOS resampling to keep text readable.
    """
    original_width, original_height = img.size

    # Calculate scaling factor to fit within target dimensions
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    return img.resize((new_width, new_height), resample=resample)


def process_directory(
    input_dir: Path,
    output_dir: Path = None,
    target_width: int = 450,
    target_height: int = 600,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Process all PNG images in directory and subdirectories.

    Args:
        input_dir: Directory containing images
        output_dir: Output directory (if None, overwrites in place)
        target_width: Target width in pixels
        target_height: Target height in pixels
        dry_run: If True, only print what would be done

    Returns:
        (processed_count, total_size_saved_mb)
    """
    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir

    # Find all PNG files
    image_files = list(input_dir.rglob("*.png"))

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return 0, 0

    print(f"Found {len(image_files)} images to process")
    print(f"Target dimensions: {target_width}x{target_height} (max, keeping aspect ratio)")

    if dry_run:
        print("\n[DRY RUN MODE - no files will be modified]\n")

    processed = 0
    total_saved_bytes = 0

    for img_path in tqdm(image_files, desc="Resizing images"):
        try:
            # Load image
            img = Image.open(img_path)
            original_size = img.size
            original_file_size = os.path.getsize(img_path)

            # Skip if already small enough
            if img.size[0] <= target_width and img.size[1] <= target_height:
                continue

            # Resize
            resized_img = resize_image(img, target_width, target_height)
            new_size = resized_img.size

            # Determine output path
            if output_dir == input_dir:
                output_path = img_path
            else:
                rel_path = img_path.relative_to(input_dir)
                output_path = output_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

            if dry_run:
                print(f"{img_path.name}: {original_size} -> {new_size}")
            else:
                # Save with high quality
                resized_img.save(output_path, "PNG", optimize=True, quality=95)
                new_file_size = os.path.getsize(output_path)
                saved_bytes = original_file_size - new_file_size
                total_saved_bytes += saved_bytes

            processed += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    total_saved_mb = total_saved_bytes / (1024 * 1024)

    print(f"\nProcessed {processed} images")
    if not dry_run:
        print(f"Total space saved: {total_saved_mb:.1f} MB")

    return processed, total_saved_mb


def main():
    parser = argparse.ArgumentParser(
        description="Resize thought images to reduce memory usage while keeping text readable"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/gsm8k",
        help="Input directory containing images (default: data/gsm8k)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite in place)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=450,
        help="Target max width in pixels (default: 450)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Target max height in pixels (default: 600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without modifying files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Thought Image Resizer")
    print("=" * 60)

    processed, saved_mb = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_width=args.width,
        target_height=args.height,
        dry_run=args.dry_run,
    )

    if processed > 0 and not args.dry_run:
        print("\n✓ Resize complete!")
        print(f"  Images will now produce ~{int(450/14) * int(600/14)} vision tokens")
        print(f"  (vs. original ~{int(900/14) * int(1160/14)} tokens)")
        print(f"  Expected memory reduction: ~75%")


if __name__ == "__main__":
    main()
