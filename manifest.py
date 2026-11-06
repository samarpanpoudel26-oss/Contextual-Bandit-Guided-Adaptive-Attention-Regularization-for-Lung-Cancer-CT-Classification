from pathlib import Path
import pandas as pd

# Define the dataset directory
DATASET_DIR = Path("Dataset")

# Store information about each image-mask pair
records = []

# Process both classification categories
for class_name, label in [("Positive", 1), ("Negative", 0)]:

    image_dir = DATASET_DIR / class_name
    mask_dir = DATASET_DIR / f"Mask_{class_name}"

    # Collect image files, excluding hidden folders
    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    for image_path in image_files:

        # Match each CT image to its generated mask
        mask_path = mask_dir / f"{image_path.stem}_mask.png"

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}: {mask_path}"
            )

        records.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "agal_mask_path": str(mask_path),
            "class": class_name,
            "label": label
        })

    print(f"{class_name}: {len(image_files)} image-mask pairs")

# Save the reconstructed manifest
df = pd.DataFrame(records)

manifest_path = DATASET_DIR / "mask_manifest.csv"
df.to_csv(manifest_path, index=False)

print("\nManifest created successfully!")
print("Total image-mask pairs:", len(df))
print("Saved to:", manifest_path)