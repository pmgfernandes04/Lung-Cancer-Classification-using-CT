#!/usr/bin/env python3
"""
feature_extraction_3d.py

This script performs 3D radiomic feature extraction on LIDC-IDRI CT scans using PyRadiomics.
It processes each scan, extracts nodules, creates 3D masks, and computes radiomic features.
The results are saved into a CSV file named 'metadata_3d.csv' for further analysis.
"""

import argparse
import gc
import logging
from pathlib import Path
from statistics import median
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pylidc as pl
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pylidc.utils import consensus
from radiomics import featureextractor
from tqdm import tqdm


# Constants
DEFAULT_OUTPUT_DIR = Path("Metadata")
DEFAULT_LOG_FILE = "feature_extraction_3d.log"
DEFAULT_CSV_FILE = DEFAULT_OUTPUT_DIR / "metadata_3d.csv"
DEFAULT_SCAN_LIMIT = 50  # Number of scans to process; set to None to process all scans
DEFAULT_CONSENSUS_LEVEL = 0.5  # Confidence level for consensus mask creation


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="3D Radiomic Feature Extraction on LIDC-IDRI CT Scans")

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output CSV file."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Path to the log file."
    )
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=DEFAULT_SCAN_LIMIT,
        help="Number of scans to process. Set to 0 or negative to process all scans."
    )
    parser.add_argument(
        "--consensus_level",
        type=float,
        default=DEFAULT_CONSENSUS_LEVEL,
        help="Confidence level for consensus mask creation."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker threads."
    )

    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    """
    Configure logging settings.

    Parameters:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO  # Set to INFO level; change to DEBUG for more verbosity
    )
    logging.info("Logging is set up.")


def inspect_scan(scan: pl.Scan) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """
    Inspect scan details without visualizing slices.

    Parameters:
        scan (pl.Scan): The scan object to inspect.

    Returns:
        Tuple containing:
            - vol (np.ndarray or None): The 3D volume array if successful, else None.
            - spacing (list or None): The spacing information if successful, else None.
    """
    pid = scan.patient_id
    logging.info(f"Inspecting scan for patient {pid}")
    print(f"Inspecting Scan for Patient ID: {pid}")

    try:
        vol = scan.to_volume()  # Extracts the 3D volume from the scan
        spacing = scan.spacings  # Retrieves spacing as [x, y, z]

        logging.info(f"Volume shape: {vol.shape}, Spacing: {spacing}")

        # Check if volume is predominantly -2048 (common air value in CT scans)
        unique_values = np.unique(vol)
        logging.debug(f"Unique pixel values in volume: {unique_values[:10]}...")

        # Log pixel value statistics
        logging.info(f"Volume pixel values range from {vol.min()} to {vol.max()}")

        return vol, spacing

    except Exception as e:
        logging.error(f"Error inspecting scan {pid}: {e}", exc_info=True)
        print(f"Error inspecting scan {pid}: {e}")
        return None, None


def calculate_malignancy(nodule_annotations: List[pl.Annotation]) -> Tuple[float, bool or str]:
    """
    Calculate the median malignancy score for a nodule and determine if it is cancerous.

    Parameters:
        nodule_annotations (List[pl.Annotation]): List of Annotation objects for the nodule.

    Returns:
        Tuple containing:
            - malignancy (float): The median malignancy score.
            - is_cancer (bool or str): True if malignant, False if benign, 'Ambiguous' otherwise.
    """
    # Extract malignancy ratings from all annotations
    malignancy_scores = [ann.malignancy for ann in nodule_annotations]
    # Calculate the median malignancy score
    malignancy = median(malignancy_scores)
    # Determine cancer status based on median score
    if malignancy >= 4:
        is_cancer = True
    elif malignancy <= 2:
        is_cancer = False
    else:
        is_cancer = 'Ambiguous'  # For a median score of 3
    return malignancy, is_cancer


def extract_features_3d(
    scan: pl.Scan,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    features_list: List[dict],
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL
) -> None:
    """
    Extract 3D radiomic features from a single scan.

    Parameters:
        scan (pl.Scan): The scan object.
        extractor (featureextractor.RadiomicsFeatureExtractor): The PyRadiomics extractor.
        features_list (List[dict]): The list to append feature dictionaries to.
        consensus_level (float): Confidence level for consensus mask creation.
    """
    pid = scan.patient_id
    logging.info(f"Processing patient {pid}")
    print(f"Processing patient {pid}")

    try:
        vol, spacing = inspect_scan(scan)
        if vol is None or spacing is None:
            logging.warning(f"Skipping scan {pid} due to inspection failure.")
            print(f"Skipping scan {pid} due to inspection failure.")
            return

        # Retrieve clustered nodules (each cluster represents a nodule)
        nodules = scan.cluster_annotations()
        if not nodules:
            logging.info(f"No nodules found for patient {pid}")
            print(f"No nodules found for patient {pid}")
            return

        for nodule_idx, nodule in enumerate(nodules):
            logging.info(f"Processing nodule {nodule_idx} for patient {pid}")
            print(f"Processing nodule {nodule_idx} for patient {pid}")

            try:
                # Generate consensus mask for the nodule
                mask_3d, cbbox, masks = consensus(nodule, clevel=consensus_level)
                logging.debug(f"Nodule {nodule_idx} bounding box: {cbbox}")

                # Crop the volume and mask to the bounding box
                vol_crop = vol[cbbox]
                mask_crop = mask_3d

                # Check if the cropped volume contains valid data (not all air)
                if np.all(vol_crop == -2048):
                    logging.warning(f"Nodule {nodule_idx} in patient {pid} has an empty cropped volume. Skipping.")
                    print(f"Nodule {nodule_idx} in patient {pid} has an empty cropped volume. Skipping.")
                    continue

                # Convert cropped volume and mask to SimpleITK images
                sitk_image = sitk.GetImageFromArray(vol_crop.astype(np.float32))
                sitk_mask = sitk.GetImageFromArray(mask_crop.astype(np.uint8))

                # Set the spacing (SimpleITK expects spacing in (x, y, z))
                sitk_image.SetSpacing(spacing[::-1])
                sitk_mask.SetSpacing(spacing[::-1])

                # Extract radiomic features using PyRadiomics
                features = extractor.execute(sitk_image, sitk_mask)

                # Calculate malignancy and cancer status
                malignancy, is_cancer = calculate_malignancy(nodule)

                # Prepare the features dictionary with necessary metadata
                features_dict = {
                    'patient_id': pid,
                    'nodule_idx': nodule_idx,
                    'malignancy': malignancy,
                    'is_cancer': is_cancer
                }

                # Add radiomic features, excluding diagnostic information
                for key in features.keys():
                    if not key.startswith('diagnostics_'):
                        features_dict[key] = features[key]

                # Append the features dictionary to the list
                features_list.append(features_dict)

                logging.info(f"Extracted features for nodule {nodule_idx} of patient {pid}")
                print(f"Extracted features for nodule {nodule_idx} of patient {pid}")

                # Free memory by deleting large objects and invoking garbage collection
                del sitk_image, sitk_mask, vol_crop, mask_crop, features
                gc.collect()

            except Exception as e:
                logging.error(
                    f"Error processing nodule {nodule_idx} for patient {pid}: {e}",
                    exc_info=True
                )
                print(f"Error processing nodule {nodule_idx} for patient {pid}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error processing scan for patient {pid}: {e}", exc_info=True)
        print(f"Error processing scan for patient {pid}: {e}")


def main():
    """Main function to perform 3D feature extraction with parallel processing and configurability."""
    # Parse command-line arguments
    args = parse_arguments()

    setup_logging(args.log_file)

    # Initialize PyRadiomics feature extractor with default settings
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()  # Enable all feature classes; consider selecting specific features if needed

    # Initialize a list to hold all feature dictionaries
    features_list: List[dict] = []

    # Define output directory and ensure it exists
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query scans from LIDC-IDRI; adjust the limit as needed
    scan_limit = args.scan_limit
    if scan_limit > 0:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).limit(scan_limit).all()
    else:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()

    if not scans:
        logging.error("No scans found in the database.")
        print("No scans found in the database.")
        return

    logging.info(f"Starting feature extraction for {len(scans)} scans.")
    print(f"Starting feature extraction for {len(scans)} scans.")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit tasks to the executor
        future_to_scan = {
            executor.submit(extract_features_3d, scan, extractor, features_list, args.consensus_level): scan
            for scan in scans
        }

        # Iterate through completed futures with a progress bar
        for future in tqdm(as_completed(future_to_scan), total=len(future_to_scan), desc="Processing Scans"):
            scan = future_to_scan[future]
            try:
                future.result()
                logging.info(f"Successfully processed scan: {scan.patient_id}")
                print(f"Successfully processed scan: {scan.patient_id}")
            except Exception as e:
                logging.error(f"Error processing scan {scan.patient_id}: {e}", exc_info=True)
                print(f"Error processing scan {scan.patient_id}: {e}")

    # Convert the list of features to a DataFrame
    df = pd.DataFrame(features_list)

    # Define the output CSV file path
    output_csv = args.csv_file

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    logging.info(f"Feature extraction completed. Results saved to {output_csv}.")
    print(f"Feature extraction completed. Results saved to {output_csv}.")
