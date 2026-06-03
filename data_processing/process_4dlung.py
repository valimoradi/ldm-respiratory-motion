"""
=============================================================================
  Step 0.1: Process raw 4D-Lung DICOM → downsampled volume tensors
=============================================================================

  Reads the already-downloaded 4D-Lung DICOM dataset, converts to HU,
  normalizes to [-1, 1], stacks slices, and resamples to 50×256×256.

  SAFE: Does NOT delete original DICOM files (unlike the built-in
  DicomProcessor.process_folder which removes originals).

  Output structure:
    data/idc_downloads/
      └─ patient_XXX/
         └─ study_XXXXXXXX/
            └─ series_DESCRIPTION_X/
                ├─ volume.pt          (50×256×256 tensor)
                └─ scan_info.json     (metadata)

  USAGE:
    python data_processing/process_4dlung.py
    python data_processing/process_4dlung.py --max-patients 5
    python data_processing/process_4dlung.py --workers 2

  RESOURCES: CPU only (no GPU), ~1-3 hours for full dataset with 4 workers.
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Configure logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
TARGET_VOL_DIMS = (50, 256, 256)
MIN_HU = -1024
MAX_HU = 3071
MIN_SLICES = 49  # Match the IDC query filter: instanceCount > 49

# Studies excluded by the original project (non-conforming)
EXCLUDED_STUDY_UIDS = {
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.170605434729890793667175785576",
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.195368562719946042143948478411",
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.983119680047871775036619601861",
}


def normalize_ct_image(image: np.ndarray) -> np.ndarray:
    """Normalize CT image to [-1, 1] range."""
    image = np.clip(image, MIN_HU, MAX_HU)
    return 2 * (image - MIN_HU) / (MAX_HU - MIN_HU) - 1


def resample_volume(vol: torch.Tensor, target: tuple) -> torch.Tensor:
    """Resample a 3D volume to target dimensions using trilinear interpolation."""
    vol_5d = vol.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    resampled = F.interpolate(vol_5d, size=target, mode="trilinear",
                               align_corners=False)
    return resampled.squeeze(0).squeeze(0)


def process_single_series(args_tuple):
    """
    Process a single DICOM series: read slices → stack → normalize → resample → save.

    This is run in a subprocess, so it must be self-contained.
    """
    series_dicom_files, output_dir, target_dims = args_tuple

    try:
        output_dir = Path(output_dir)
        volume_path = output_dir / "volume.pt"

        # Resume support: skip if volume already exists
        if volume_path.exists():
            return {"status": "skipped", "path": str(output_dir)}

        # Read DICOM headers to check for exclusions
        first_ds = pydicom.dcmread(series_dicom_files[0], stop_before_pixels=True)
        study_uid = str(getattr(first_ds, "StudyInstanceUID", ""))
        if study_uid in EXCLUDED_STUDY_UIDS:
            return {"status": "excluded", "path": str(output_dir)}

        # Read all slices
        slices = []
        positions = []
        for dcm_path in series_dicom_files:
            try:
                ds = pydicom.dcmread(dcm_path)
                if ds.pixel_array.shape != (512, 512):
                    continue

                image = ds.pixel_array.astype(np.float32)
                if ds.Modality == "CT":
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    image = image * slope + intercept
                    image = normalize_ct_image(image)

                position = float(getattr(ds, "SliceLocation", 0.0))
                slices.append(image)
                positions.append(position)
            except Exception as e:
                logger.debug(f"Skipping slice {dcm_path}: {e}")
                continue

        if len(slices) < MIN_SLICES:
            return {"status": "too_few_slices", "n_slices": len(slices),
                    "path": str(output_dir)}

        # Sort by slice position
        sorted_indices = np.argsort(positions)
        slices = [slices[i] for i in sorted_indices]

        # Stack to 3D volume [D, H, W]
        volume = torch.from_numpy(np.stack(slices, axis=0))

        # Resample to target dimensions
        if target_dims is not None:
            volume = resample_volume(volume, target_dims)

        # Save volume
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(volume, volume_path)

        # Save metadata
        metadata = extract_metadata(first_ds, len(slices), target_dims)
        with open(output_dir / "scan_info.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "ok", "shape": list(volume.shape),
                "path": str(output_dir)}

    except Exception as e:
        return {"status": "error", "error": str(e),
                "path": str(output_dir) if output_dir else "unknown"}


def extract_metadata(ds, n_slices, target_dims):
    """Extract relevant metadata from a DICOM dataset."""
    pixel_spacing = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

    meta = {
        "patient_id": str(getattr(ds, "PatientID", "unknown")),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "unknown")),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "unknown")),
        "series_description": str(getattr(ds, "SeriesDescription", "unknown")),
        "modality": str(getattr(ds, "Modality", "unknown")),
        "n_slices": n_slices,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "original_image_size": [512, 512],
        "original_volume_size": [n_slices, 512, 512],
    }

    if target_dims:
        # Compute resampled spacing
        orig_z, orig_y, orig_x = n_slices, 512, 512
        tgt_z, tgt_y, tgt_x = target_dims
        meta["resampled_pixel_spacing"] = [
            pixel_spacing[0] * orig_y / tgt_y,
            pixel_spacing[1] * orig_x / tgt_x,
        ]
        meta["resampled_slice_thickness"] = slice_thickness * orig_z / tgt_z
        meta["resampled_volume_size"] = list(target_dims)

    return meta


def discover_series(raw_data_path: str, max_patients: int = None):
    """
    Discover all DICOM series from the raw 4D-Lung dataset.

    Groups DICOM files by patient → study → series using headers.

    Returns:
        list of (series_files, output_dir) tuples
    """
    raw_path = Path(raw_data_path)

    # The 4D-Lung data is organized: 4D-Lung/100_HM.../STUDY_UID/SERIES_UID/
    lung_dir = raw_path / "4D-Lung"
    if not lung_dir.exists():
        # Maybe the path already points to the 4D-Lung directory
        lung_dir = raw_path

    patient_dirs = sorted([d for d in lung_dir.iterdir() if d.is_dir()])
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]

    logger.info(f"Found {len(patient_dirs)} patient directories")

    all_series = []
    for patient_dir in tqdm(patient_dirs, desc="Discovering patients"):
        patient_id = patient_dir.name.split("_")[0]  # e.g., "100"

        # Walk through study/series directories
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            study_uid_short = study_dir.name[-8:].replace(".", "_")

            for series_dir in sorted(study_dir.iterdir()):
                if not series_dir.is_dir():
                    continue

                # Collect all DICOM files in this series
                dcm_files = list(series_dir.rglob("*.dcm"))
                if len(dcm_files) < MIN_SLICES:
                    continue

                # Try to get series description from first file
                try:
                    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                    series_desc = str(
                        getattr(ds, "SeriesDescription", "unknown")
                    ).replace(" ", "_")
                    series_uid_short = str(
                        getattr(ds, "SeriesInstanceUID", "unknown")
                    )[-8:].replace(".", "_")
                except Exception:
                    series_desc = "unknown"
                    series_uid_short = series_dir.name[-8:].replace(".", "_")

                output_dir = (
                    f"patient_{patient_id}/"
                    f"study_{study_uid_short}/"
                    f"series_{series_desc}_{series_uid_short}"
                )

                all_series.append((
                    [str(f) for f in dcm_files],
                    output_dir,
                ))

    logger.info(f"Discovered {len(all_series)} series across "
                f"{len(patient_dirs)} patients")
    return all_series


def main():
    parser = argparse.ArgumentParser(
        description="Process raw 4D-Lung DICOM to volume tensors (Step 0.1)")
    parser.add_argument("--raw-data", type=str,
                        default=r"E:\Cancer\datasets\4dct\manifest-ObLxS9Wd1073675925233948759",
                        help="Path to raw 4D-Lung data")
    parser.add_argument("--output", type=str,
                        default=r"E:\Cancer\curse_of_optimization\motion\ldm-respiratory-motion\data\idc_downloads",
                        help="Output directory for processed volumes")
    parser.add_argument("--target-dims", type=str, default="50,256,256",
                        help="Target volume dimensions (D,H,W)")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Limit to first N patients (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    target_dims = tuple(int(x) for x in args.target_dims.split(","))

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Step 0.1: DICOM → Volume Processing (4D-Lung)             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Raw data:   {args.raw_data:<46s}║
    ║  Output:     {args.output:<46s}║
    ║  Target:     {str(target_dims):<46s}║
    ║  Workers:    {args.workers:<46d}║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Discover series ──────────────────────────────────────────
    t0 = time.time()
    all_series = discover_series(args.raw_data, args.max_patients)

    if not all_series:
        logger.error("No DICOM series found!")
        return

    # ── Prepare tasks ────────────────────────────────────────────
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dcm_files, rel_output in all_series:
        full_output = str(output_base / rel_output)
        tasks.append((dcm_files, full_output, target_dims))

    # ── Process in parallel ──────────────────────────────────────
    logger.info(f"Processing {len(tasks)} series with {args.workers} workers ...")

    results = {"ok": 0, "skipped": 0, "excluded": 0,
               "too_few_slices": 0, "error": 0}

    # Save progress log
    log_path = output_base / "processing_log.json"
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_series, t): t
                   for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing series"):
            try:
                result = future.result(timeout=300)  # 5min timeout
                status = result.get("status", "error")
                results[status] = results.get(status, 0) + 1

                if status == "error":
                    errors.append(result)
                    logger.warning(f"Error: {result.get('error', 'unknown')} "
                                   f"at {result.get('path', '?')}")
            except Exception as e:
                results["error"] += 1
                errors.append({"status": "exception", "error": str(e)})

    total_time = time.time() - t0

    # ── Save processing log ──────────────────────────────────────
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": total_time,
        "results": results,
        "errors": errors[:50],  # Limit error list
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  STEP 0.1 COMPLETE ({total_time/60:.1f} minutes)")
    print(f"{'=' * 60}")
    print(f"  Processed:  {results['ok']}")
    print(f"  Skipped:    {results['skipped']} (already done)")
    print(f"  Excluded:   {results['excluded']}")
    print(f"  Too few:    {results['too_few_slices']}")
    print(f"  Errors:     {results['error']}")
    print(f"  Log saved:  {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
