"""Generate SegResNet lung pseudo-masks and a fixed AGAL screening manifest.

Usage:
    python generate_lung_masks.py --dataset Dataset

Optional checkpoint override (if repository contains multiple checkpoints):
    python generate_lung_masks.py --dataset Dataset --checkpoint path/in/repo.pt

Requires: pip install torch monai huggingface_hub opencv-python numpy pandas

Note: The checkpoint was trained for its own image distribution. These masks are
pseudo-labels, not clinically validated lung segmentations.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from monai.networks.nets import SegResNet

REPO_ID = "HalmosiL/roi-segresnet-2d"
IMAGE_SIZE = (256, 256)
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device, checkpoint_name=None):
    """Load the 2D SegResNet architecture and pretrained checkpoint strictly."""
    model = SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_filters=16,
        blocks_down=(1, 2, 2, 4, 4),
        blocks_up=(1, 1, 1, 1),
        dropout_prob=0.1,
    )

    if checkpoint_name is None:
        candidates = [
            f for f in list_repo_files(REPO_ID)
            if f.lower().endswith((".pt", ".pth", ".ckpt"))
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Found {len(candidates)} checkpoints: {candidates}. "
                "Pass --checkpoint with the exact filename used previously."
            )
        checkpoint_name = candidates[0]

    path = hf_hub_download(repo_id=REPO_ID, filename=checkpoint_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state dictionary")

    checkpoint = {
        k.removeprefix("module."): v for k, v in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_name}")
    return model, checkpoint_name


def clean_mask_v3(mask, border_width=5):
    """Remove border-touching components and wide inferior artifacts.

    This reproduces the tested V3 heuristic; it can remove true lung pixels.
    Input and output values are binary 0/1.
    """
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    cleaned = binary.copy()
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    for component_id in range(1, n):
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        cw = stats[component_id, cv2.CC_STAT_WIDTH]
        ch = stats[component_id, cv2.CC_STAT_HEIGHT]
        center_y = centroids[component_id][1]

        touches_border = (
            x < border_width or y < border_width
            or x + cw > w - border_width
            or y + ch > h - border_width
        )
        bottom_artifact = (
            center_y > 0.75 * h
            and cw > 2.0 * ch
            and cw > 0.25 * w
        )
        if touches_border or bottom_artifact:
            cleaned[labels == component_id] = 0

    return cleaned


def approve_mask(mask, min_coverage=0.03, max_coverage=0.65):
    """Heuristic AGAL screening, not anatomical verification."""
    binary = (mask > 0).astype(np.uint8)
    coverage = float(binary.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 0.01 * binary.size
    significant = sum(
        stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)
    )

    if coverage == 0:
        status = "empty"
    elif coverage < min_coverage:
        status = "too_small"
    elif coverage > max_coverage:
        status = "too_large"
    elif significant == 0:
        status = "fragmented"
    else:
        status = "provisionally_approved"

    return {
        "agal_approved": int(status == "provisionally_approved"),
        "quality_status": status,
        "coverage": coverage,
        "components": int(significant),
    }


def run(dataset, checkpoint_name=None, overwrite=False):
    manifest_path = dataset / "mask_manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest exists: {manifest_path}. Use --overwrite only if intended."
        )
    device = choose_device()
    print("Device:", device)
    model, selected_checkpoint = load_model(device, checkpoint_name)
    records = []

    for class_name, label in (("Positive", 1), ("Negative", 0)):
        image_dir = dataset / class_name
        mask_dir = dataset / f"Mask_{class_name}"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        print(f"\n{class_name}: {len(image_files)} images")

        for i, image_path in enumerate(image_files, start=1):
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output exists: {mask_path}. Use --overwrite only if intended."
                )

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(
                resized.astype(np.float32) / 255.0
            )[None, None].to(device)

            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            raw_mask = (probabilities >= 0.5).astype(np.uint8)
            # Initial conservative cleanup used in the earlier notebook
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                raw_mask, connectivity=8
            )
            initial_mask = np.zeros_like(raw_mask)
            for component_id in range(1, n):
                if stats[component_id, cv2.CC_STAT_AREA] >= 30:
                    initial_mask[labels == component_id] = 1

            # Match the earlier pipeline: resize initial mask first, then apply V3
            # to the original-resolution mask (V3 uses relative geometry).
            h, w = image.shape
            original_resolution = cv2.resize(
                initial_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            cleaned = clean_mask_v3(original_resolution)
            final_png = cleaned.astype(np.uint8) * 255

            if not cv2.imwrite(str(mask_path), final_png):
                raise IOError(f"Could not write mask: {mask_path}")

            screening = approve_mask(cleaned)
            records.append({
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "class": class_name,
                "label": label,
                "agal_mask_path": str(mask_path),
                "mask_coverage_before_v3": float(original_resolution.mean()),
                "clean_coverage": float(cleaned.mean()),
                "checkpoint": selected_checkpoint,
                **screening,
            })
            if i % 100 == 0 or i == len(image_files):
                print(f"  {i}/{len(image_files)}")

    df = pd.DataFrame(records)
    manifest_path = dataset / "mask_manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest exists: {manifest_path}. Use --overwrite if intended."
        )
    df.to_csv(manifest_path, index=False)
    print("\nGenerated:", len(df), "masks")
    print("\nQuality screening:\n", df["quality_status"].value_counts())
    print("\nApproval by class:\n", pd.crosstab(df["class"], df["agal_approved"]))
    print("\nManifest:", manifest_path)
    



