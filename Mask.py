import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm





def generate_lung_mask(image_path, output_dir=None, visualize=False):
    """
    Generate a lung mask from CT image using thresholding and morphological operations.
    
    Args:
        image_path: Path to CT image
        output_dir: Directory to save mask (optional)
        visualize: Whether to display results
    
    Returns:
        mask: Binary lung mask
        original: Original image
    """
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original = img.copy()
    
    # Step 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Step 3: Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 4: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise with opening
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill small holes with closing
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 5: Find largest contours (lung regions)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from scratch
    mask = np.zeros_like(closed)
    
    # Keep only the two largest contours (left and right lungs)
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Draw top 2 largest contours (lungs)
        for i in range(min(2, len(contours))):
            cv2.drawContours(mask, [contours[i]], 0, 255, -1)
    
    # Step 6: Additional morphological smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    
    # Save mask if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / (Path(image_path).stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)
    
    return mask, original


def calculate_lung_coverage(mask):
    """Calculate lung region coverage as percentage."""
    total_pixels = mask.size
    lung_pixels = cv2.countNonZero(mask)
    coverage_percent = (lung_pixels / total_pixels) * 100
    return coverage_percent, lung_pixels


def batch_process_images(input_dir, output_dir, num_images=1391, visualize_first=True):
    """
    Process first N images from input directory and generate lung masks.
    
    Args:
        input_dir: Directory containing CT images
        output_dir: Directory to save masks
        num_images: Number of images to process (default 10)
        visualize_first: Whether to create visualization grid
    
    Returns:
        DataFrame with statistics for all processed images
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    masks_dir = output_path / "masks"
    masked_images_dir = output_path / "masked_images"
    masks_dir.mkdir(exist_ok=True)
    masked_images_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Sort and take first N images
    image_files = sorted(image_files)[:num_images]
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return None
    
    print(f"Found {len(image_files)} images. Processing first {min(num_images, len(image_files))} images...")
    
    stats_list = []
    processed_images = []
    masks_list = []
    
    # Process each image
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Generate mask
            mask, original = generate_lung_mask(str(image_file), masks_dir, visualize=False)
            
            # Calculate statistics
            coverage, lung_pixels = calculate_lung_coverage(mask)
            
            # Apply mask to original image
            masked_image = cv2.bitwise_and(original, original, mask=mask)
            
            # Save masked image
            masked_path = masked_images_dir / (image_file.stem + "_masked.png")
            cv2.imwrite(str(masked_path), masked_image)
            
            # Store statistics
            stats_list.append({
                'Image_ID': idx + 1,
                'Filename': image_file.name,
                'Coverage_%': round(coverage, 2),
                'Lung_Pixels': lung_pixels,
                'Total_Pixels': mask.size,
                'Image_Shape': f"{original.shape[0]}x{original.shape[1]}"
            })
            
            processed_images.append(original)
            masks_list.append(mask)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(stats_list)
    
    # Save statistics
    stats_path = output_path / "statistics.csv"
    df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved to: {stats_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    if len(df) > 0:
        print(df.to_string(index=False))
        print("="*70)
        print(f"\nMean Lung Coverage: {df['Coverage_%'].mean():.2f}%")
        print(f"Std Dev Coverage: {df['Coverage_%'].std():.2f}%")
        print(f"Min Coverage: {df['Coverage_%'].min():.2f}%")
        print(f"Max Coverage: {df['Coverage_%'].max():.2f}%")
    else:
        print("No images processed successfully")
    
    # Create visualization grid
    if visualize_first and len(processed_images) > 0:
        create_comparison_grid(processed_images, masks_list, output_path)
    
    # Save individual visualizations for first 3 images
    if len(processed_images) >= 1:
        save_detailed_visualizations(processed_images[:3], masks_list[:3], output_path)
    
    print(f"\nProcessed {len(processed_images)} images successfully")
    print(f"Masks saved to: {masks_dir}")
    print(f"Masked images saved to: {masked_images_dir}")
    
    return df


def create_comparison_grid(images, masks, output_dir):
    """Create a comparison grid of original images and masks."""
    
    n_images = len(images)
    n_cols = min(5, n_images)  # Max 5 columns
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    idx = 0
    for i in range(n_images):
        # Original image
        axes[idx].imshow(images[i], cmap='gray')
        axes[idx].set_title(f'Image {i+1}')
        axes[idx].axis('off')
        idx += 1
        
        # Mask
        axes[idx].imshow(masks[i], cmap='gray')
        axes[idx].set_title(f'Mask {i+1}')
        axes[idx].axis('off')
        idx += 1
    
    # Hide remaining subplots
    while idx < len(axes):
        axes[idx].axis('off')
        idx += 1
    
    plt.tight_layout()
    save_path = output_dir / "comparison_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison grid saved to: {save_path}")
    plt.close()


def save_detailed_visualizations(images, masks, output_dir):
    """Save detailed step-by-step visualizations for first few images."""
    
    for img_idx, (image, mask) in enumerate(zip(images, masks)):
        # Recreate intermediate steps for visualization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Original Image {img_idx+1}')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('CLAHE Enhanced')
        axes[1].axis('off')
        
        axes[2].imshow(thresh, cmap='gray')
        axes[2].set_title('Otsu Threshold')
        axes[2].axis('off')
        
        axes[3].imshow(mask, cmap='gray')
        axes[3].set_title('Final Mask')
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f"detailed_image_{img_idx+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Detailed visualization saved: {save_path}")


if __name__ == "__main__":
    # ============================================
    # CONFIGURATION - MODIFY THIS SECTION
    # ============================================
    
    # Option 1: Process from uploaded sample
    INPUT_DIR = "/Users/samarpanpoudel/Desktop/Loss Function/Dataset/Positive"  # Your input directory with images
    OUTPUT_DIR = "/Users/samarpanpoudel/Desktop/Loss Function/Dataset/Mask_Positive"
    NUM_IMAGES = 1391 
    
    # Option 2: If you have images in a different directory, change INPUT_DIR:
    # INPUT_DIR = "/path/to/your/ct/images"
    
    # ============================================
    # RUN BATCH PROCESSING
    # ============================================
    
    print(f"\n{'='*70}")
    print(f"BATCH LUNG MASK GENERATION")
    print(f"{'='*70}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Processing: First {NUM_IMAGES} images")
    print(f"{'='*70}\n")
    
    # Run batch processing
    stats_df = batch_process_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_images=NUM_IMAGES,
        visualize_first=True
    )
    
    if stats_df is not None:
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)