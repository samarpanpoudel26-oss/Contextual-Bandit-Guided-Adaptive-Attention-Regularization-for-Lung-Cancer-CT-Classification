"""Audit existing lung pseudo-masks and record AGAL eligibility.

Does not generate, modify, or delete image/mask files.

Usage:
    python mask_quality.py --dataset Dataset
    python mask_quality.py --dataset Dataset --reject-file Dataset/manual_rejections.csv

Optional rejection CSV: a column named image_path OR mask_path, containing
paths or filenames to reject (prefer full paths to avoid ambiguous names).

Thresholds are heuristic; approval is NOT anatomical validation.
"""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


def resolve_existing_path(value, dataset, manifest_parent):
    """Resolve saved paths from either project root or Dataset directory."""
    if pd.isna(value) or not str(value).strip():
        return None
    p = Path(str(value))
    candidates = [p] if p.is_absolute() else [
        p, manifest_parent / p, manifest_parent.parent / p,
        dataset / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def measure_mask(path, min_coverage=0.03, max_coverage=0.65,
                 min_component_fraction=0.01, border_width=5):
    """Compute screening features on a saved binary mask (no modification)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    binary = (mask > 127).astype(np.uint8)
    h, w = binary.shape
    coverage = float(binary.mean())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=int)
    substantial = int(np.count_nonzero(areas >= min_component_fraction * binary.size))
    component_count = int(n - 1)
    largest_fraction = float(areas.max() / max(1, areas.sum())) if len(areas) else 0.0

    # Fraction of border pixels labeled as lung. This is a warning, not a rejection.
    b = max(1, min(border_width, h // 2, w // 2))
    border = np.zeros((h, w), dtype=bool)
    border[:b] = True
    border[-b:] = True
    border[:, :b] = True
    border[:, -b:] = True
    border_fraction = float(binary[border].mean())
    border_flag = int(border_fraction > 0.01)

    if coverage == 0:
        status = "empty"
    elif coverage < min_coverage:
        status = "too_small"
    elif coverage > max_coverage:
        status = "too_large"
    elif substantial == 0:
        status = "fragmented"
    else:
        status = "provisionally_approved"

    return {
        "mask_coverage": coverage,
        "component_count": component_count,
        "significant_components": substantial,
        "largest_component_fraction": largest_fraction,
        "border_fraction": border_fraction,
        "border_flag": border_flag,
        "quality_status": status,
        "agal_approved": int(status == "provisionally_approved"),
    }


def load_manual_rejections(path):
    if path is None:
        return set()
    reject_df = pd.read_csv(path)
    cols = [c for c in ("image_path", "mask_path") if c in reject_df.columns]
    if not cols:
        raise ValueError("Rejection CSV needs an image_path or mask_path column")
    entries = set()
    for col in cols:
        entries.update(str(x).strip() for x in reject_df[col].dropna() if str(x).strip())
    return entries


def is_rejected(row, rejected):
    """Support full paths or filenames; ambiguous filenames reject all matches."""
    for col in ("image_path", "agal_mask_path"):
        value = str(row[col])
        if value in rejected or Path(value).name in rejected:
            return True
    return False


def run(dataset, reject_file=None, min_coverage=0.03, max_coverage=0.65):
    dataset = Path(dataset)
    manifest = dataset / "mask_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    df = pd.read_csv(manifest)
    if "image_path" not in df.columns:
        raise ValueError("Manifest must include image_path")
    if "agal_mask_path" not in df.columns:
        if "mask_path" in df.columns:
            df["agal_mask_path"] = df["mask_path"]
        else:
            raise ValueError("Manifest must include agal_mask_path or mask_path")
    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image_path entries; fix before quality assessment")

    rejected = load_manual_rejections(reject_file)
    results = []
    for _, row in df.iterrows():
        image = resolve_existing_path(row["image_path"], dataset, manifest.parent)
        mask = resolve_existing_path(row["agal_mask_path"], dataset, manifest.parent)
        if image is None:
            raise FileNotFoundError(f"Missing CT image: {row['image_path']}")
        if mask is None:
            raise FileNotFoundError(f"Missing lung mask: {row['agal_mask_path']}")
        stats = measure_mask(mask, min_coverage, max_coverage)
        manual = int(is_rejected(row, rejected))
        stats["manual_reject"] = manual
        if manual:
            stats["quality_status"] = "manually_rejected"
            stats["agal_approved"] = 0
        results.append(stats)

    # Replace prior quality fields, retaining source and generation metadata.
    quality = pd.DataFrame(results)
    df = df.drop(columns=list(quality.columns), errors="ignore")
    df = pd.concat([df.reset_index(drop=True), quality], axis=1)

    # Keep a backup of the previous manifest before writing updated quality data.
    backup = manifest.with_name("mask_manifest_before_quality.csv")
    if not backup.exists():
        shutil.copy2(manifest, backup)
    temp = manifest.with_name("mask_manifest.tmp.csv")
    df.to_csv(temp, index=False)
    temp.replace(manifest)

    print(f"Images retained: {len(df)}")
    print(f"AGAL eligible (provisional): {int(df['agal_approved'].sum())}")
    print(f"AGAL ineligible: {int((df['agal_approved'] == 0).sum())}")
    print("\nQuality status:\n", df["quality_status"].value_counts().to_string())
    print("\nBorder warnings (not automatic rejection):", int(df["border_flag"].sum()))
    if "class" in df.columns:
        print("\nBy class:\n", pd.crosstab(df["class"], df["agal_approved"]).to_string())
    print(f"\nUpdated: {manifest}\nBackup: {backup}")
    print("Reminder: automated screening does not establish anatomical correctness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--reject-file", type=Path, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.03)
    parser.add_argument("--max-coverage", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.min_coverage < args.max_coverage <= 1:
        parser.error("Coverage thresholds must satisfy 0 <= min < max <= 1")
    run(args.dataset, args.reject_file, args.min_coverage, args.max_coverage)
