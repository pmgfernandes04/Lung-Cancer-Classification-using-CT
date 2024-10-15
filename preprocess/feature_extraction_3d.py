#!/usr/bin/env python3
"""
feature_extraction_3d.py

This script performs 3D radiomic and CNN-based feature extraction on LIDC-IDRI CT scans using PyRadiomics and a pre-trained 3D ResNet from MONAI.
It processes each scan, extracts nodules, creates 3D masks, preprocesses the nodules, extracts features,
and computes radiomic and CNN features. The results, along with labels, are saved into two CSV files named 'meta_info.csv' for radiomic features and 'cnn_features.csv' for CNN features for further analysis.
"""

# ===============================
#         IMPORTS
# ===============================

import argparse
import gc
import logging
from pathlib import Path
from statistics import median
from typing import List, Tuple, Optional
import threading  # For thread-local storage to handle models in multi-threading

import numpy as np
import pandas as pd
import pylidc as pl  # Library for accessing LIDC-IDRI dataset
import SimpleITK as sitk  # For image processing and handling medical images
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from pylidc.utils import consensus  # To create consensus masks from multiple annotations
from radiomics import featureextractor  # PyRadiomics for radiomic feature extraction
from tqdm import tqdm  # For progress bars

import torch
import torch.nn as nn

# Import MONAI's pre-trained ResNet18 for CNN-based feature extraction
from monai.networks.nets import resnet18

from scipy.ndimage import zoom  # For resampling images

# ===============================
#         CONSTANTS
# ===============================

# Default directory to save output CSV files
DEFAULT_OUTPUT_DIR = Path("Metadata")

# Default log file path
DEFAULT_LOG_FILE = "feature_extraction_3d.log"

# Default scan limit: number of scans to process; set to 0 or negative to process all scans
DEFAULT_SCAN_LIMIT = 50

# Default confidence level for consensus mask creation
DEFAULT_CONSENSUS_LEVEL = 0.5

# Size of the crop in millimeters for resizing nodules
CROP_SIZE_MM = 32

# Device configuration: Use GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
#         ARGUMENT PARSING
# ===============================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to customize script behavior.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="3D Radiomic and CNN Feature Extraction on LIDC-IDRI CT Scans"
    )

    # Directory to save output CSV files
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output CSV files."
    )

    # Path to the log file
    parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Path to the log file."
    )

    # Path to save the original output CSV file with all features (optional, since we're saving separate CSVs)
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=None,  # Not used in the current implementation
        help="Path to save the original output CSV file with all features."
    )

    # Number of scans to process; set to 0 or negative to process all scans
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=DEFAULT_SCAN_LIMIT,
        help="Number of scans to process. Set to 0 or negative to process all scans."
    )

    # Confidence level for consensus mask creation (between 0 and 1)
    parser.add_argument(
        "--consensus_level",
        type=float,
        default=DEFAULT_CONSENSUS_LEVEL,
        help="Confidence level for consensus mask creation."
    )

    # Number of parallel worker threads; reduced to mitigate Out-Of-Memory (OOM) issues
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker threads."
    )

    # Dimension of the CNN feature vector
    parser.add_argument(
        "--cnn_feature_dim",
        type=int,
        default=128,
        help="Dimension of the CNN feature vector."
    )

    # Path to a pre-trained CNN model; if not provided, a MONAI pre-trained model will be used
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a pre-trained CNN model. If not provided, a MONAI pre-trained model will be used."
    )

    return parser.parse_args()

# ===============================
#         LOGGING SETUP
# ===============================

def setup_logging(log_file: str) -> None:
    """
    Configure logging settings to track script execution and debug information.

    Parameters:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,  # Log file path
        filemode='a',  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        level=logging.INFO  # Logging level (INFO for general messages)
    )
    logging.info("Logging is set up.")  # Initial log entry
    print("Logging is set up.")  # Print to console for immediate feedback

# ===============================
#         UTILITY FUNCTIONS
# ===============================

def resample_scan(
    volume: np.ndarray,
    original_spacing: List[float],
    new_spacing: List[float] = [1.0, 1.0, 1.0]
) -> Tuple[np.ndarray, List[float]]:
    """
    Resample the 3D scan to new voxel spacing using trilinear interpolation.

    Parameters:
        volume (np.ndarray): The original 3D volume with shape (z, y, x).
        original_spacing (List[float]): Original voxel spacing in mm for each axis [x, y, z].
        new_spacing (List[float], optional): Desired voxel spacing in mm for each axis [x, y, z]. Defaults to [1.0, 1.0, 1.0].

    Returns:
        Tuple[np.ndarray, List[float]]: Resampled 3D volume and the new spacing used.
    """
    # Calculate resize factors for each axis
    resize_factor = np.array(original_spacing) / np.array(new_spacing)
    # Apply trilinear interpolation to resample the volume
    resampled_volume = zoom(volume, resize_factor, order=1)  # order=1 -> trilinear
    return resampled_volume, new_spacing


def clamp_hounsfield_units(
    volume: np.ndarray,
    min_hu: float = -1000.0,
    max_hu: float = 400.0
) -> np.ndarray:
    """
    Clamp the Hounsfield Unit (HU) values of the CT scan to a specified range.

    Parameters:
        volume (np.ndarray): The 3D volume with HU values.
        min_hu (float, optional): Minimum HU value to clamp. Defaults to -1000.0.
        max_hu (float, optional): Maximum HU value to clamp. Defaults to 400.0.

    Returns:
        np.ndarray: Clamped volume with HU values within [min_hu, max_hu].
    """
    return np.clip(volume, min_hu, max_hu)


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity values of the volume to have zero mean and unit variance.

    Parameters:
        volume (np.ndarray): The 3D volume with normalized HU values.

    Returns:
        np.ndarray: Intensity-normalized volume.
    """
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        # Avoid division by zero; return zero-centered data
        return volume - mean
    return (volume - mean) / std


def get_bounding_box(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    """
    Determine the bounding box of a binary mask.

    Parameters:
        mask (np.ndarray): Binary mask of the nodule with shape (z, y, x).

    Returns:
        Tuple[slice, slice, slice]: Slices defining the bounding box for z, y, and x axes.
                                   Returns (slice(0,0), slice(0,0), slice(0,0)) if the mask is empty.
    """
    coords = np.argwhere(mask)  # Find coordinates where mask is True
    if coords.size == 0:
        # Empty mask; return empty slices
        return (slice(0, 0), slice(0, 0), slice(0, 0))
    # Determine minimum and maximum indices along each axis
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1  # Add 1 to include the max index
    return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))


def extract_and_resize_nodule(
    volume: np.ndarray,
    mask: np.ndarray,
    crop_size_mm: int = 32
) -> np.ndarray:
    """
    Extract the nodule from the volume using the mask and resize it to a fixed size.

    Parameters:
        volume (np.ndarray): The 3D normalized volume with shape (z, y, x).
        mask (np.ndarray): Binary mask of the nodule with shape (z, y, x).
        crop_size_mm (int, optional): Desired size of the crop in mm for each axis. Defaults to 32.

    Returns:
        np.ndarray: Resized nodule volume with shape (crop_size_mm, crop_size_mm, crop_size_mm).
                    Returns a zero-filled array if the mask is empty.
    """
    bbox = get_bounding_box(mask)  # Get bounding box slices
    if bbox == (slice(0, 0), slice(0, 0), slice(0, 0)):
        # Empty mask; return a zero-filled array
        return np.zeros((crop_size_mm, crop_size_mm, crop_size_mm), dtype=np.float32)

    # Crop the volume and mask to the bounding box
    cropped_vol = volume[bbox]
    cropped_mask = mask[bbox]

    # Apply the mask to the cropped volume to isolate the nodule
    nodule_vol = cropped_vol * cropped_mask

    # Define the desired output shape for the resized nodule
    desired_shape = (crop_size_mm, crop_size_mm, crop_size_mm)

    # Current shape of the nodule volume
    current_shape = nodule_vol.shape

    # Calculate zoom factors for each axis to achieve the desired shape
    zoom_factors = [d / c for d, c in zip(desired_shape, current_shape)]

    # Resize the nodule volume using trilinear interpolation
    resized_nodule = zoom(nodule_vol, zoom_factors, order=1)  # order=1 -> trilinear

    return resized_nodule.astype(np.float32)


def normalize_nodule(volume: np.ndarray) -> np.ndarray:
    """
    Normalize the nodule volume to have zero mean and unit variance.

    Parameters:
        volume (np.ndarray): The 3D nodule volume.

    Returns:
        np.ndarray: Normalized nodule volume.
    """
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        # Avoid division by zero; return zero-centered data
        return volume - mean
    return (volume - mean) / std


def calculate_malignancy(
    nodule_annotations: List[pl.Annotation]
) -> Tuple[float, bool or str]:
    """
    Calculate the median malignancy score for a nodule and determine its cancer status.

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

# ===============================
#         MODEL HANDLING
# ===============================

# Thread-local storage to hold model instances per thread, ensuring thread safety
thread_local = threading.local()


def get_cnn_model(args: argparse.Namespace) -> nn.Module:
    """
    Retrieve the CNN model for the current thread, initializing it if necessary.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments containing model configuration.

    Returns:
        nn.Module: The CNN model instance ready for feature extraction.
    """
    if not hasattr(thread_local, 'cnn_model'):
        # Initialize the pre-trained ResNet18 model from MONAI with spatial_dims=3 for 3D convolutions
        cnn_model = resnet18(
            pretrained=True,            # Use pre-trained weights
            spatial_dims=3,             # 3D convolutions
            n_input_channels=1,         # Single-channel CT images
            feed_forward=False,         # Exclude the final classification layer
            num_classes=args.cnn_feature_dim,  # Output feature dimension
            shortcut_type='A',          # Required for MedicalNet pre-trained models
            bias_downsample=False       # Whether to include bias in downsampling layers
        ).to(DEVICE)  # Move model to the configured device (GPU or CPU)

        if args.model_path:
            # Load pre-trained model weights from the specified path
            cnn_model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
            logging.info(f"Loaded CNN model weights from {args.model_path}")
            print(f"Loaded CNN model weights from {args.model_path}")

        cnn_model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
        thread_local.cnn_model = cnn_model  # Store the model in thread-local storage

    return thread_local.cnn_model  # Return the model instance

# ===============================
#         FEATURE EXTRACTION
# ===============================

def extract_features_3d(
    scan: pl.Scan,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    args: argparse.Namespace
) -> Tuple[List[dict], List[dict]]:
    """
    Extract 3D radiomic and CNN features from a single CT scan.

    Parameters:
        scan (pl.Scan): The scan object representing a single patient's CT scan.
        extractor (featureextractor.RadiomicsFeatureExtractor): The PyRadiomics extractor for radiomic features.
        args (argparse.Namespace): Parsed command-line arguments containing configuration parameters.

    Returns:
        Tuple[List[dict], List[dict]]: Two lists containing radiomic feature dictionaries and CNN feature dictionaries for each nodule.
    """
    radiomic_features_list = []  # List to store radiomic features for all nodules in this scan
    cnn_features_list = []       # List to store CNN features for all nodules in this scan
    pid = scan.patient_id         # Patient ID for logging and identification

    logging.info(f"Processing patient {pid}")  # Log the start of processing for this patient
    print(f"Processing patient {pid}")        # Print to console for immediate feedback

    try:
        # Extract the 3D volume data from the scan
        vol = scan.to_volume()  # Shape: (z, y, x)

        # Retrieve the original voxel spacing in mm for each axis
        original_spacing = scan.spacings  # [x, y, z]

        logging.info(f"Original volume shape: {vol.shape}, Original spacing: {original_spacing}")

        # Resample the scan to isotropic spacing (1x1x1 mm³) using trilinear interpolation
        vol_resampled, new_spacing = resample_scan(vol, original_spacing, new_spacing=[1.0, 1.0, 1.0])
        logging.info(f"Resampled volume shape: {vol_resampled.shape}, New spacing: {new_spacing}")

        # Clamp the Hounsfield Units to the specified range to reduce noise and focus on relevant intensities
        vol_clamped = clamp_hounsfield_units(vol_resampled, min_hu=-1000.0, max_hu=400.0)

        # Normalize the intensity values to have zero mean and unit variance for standardized feature extraction
        vol_normalized = normalize_intensity(vol_clamped)

        # Cluster annotations to identify unique nodules; each cluster represents a single nodule
        nodules = scan.cluster_annotations()
        if not nodules:
            # No nodules found for this patient; log and skip
            logging.info(f"No nodules found for patient {pid}")
            print(f"No nodules found for patient {pid}")
            return radiomic_features_list, cnn_features_list

        # Retrieve the CNN model for the current thread to ensure thread safety
        cnn_model = get_cnn_model(args)

        # Iterate over each nodule in the scan
        for nodule_idx, nodule in enumerate(nodules):
            logging.info(f"Processing nodule {nodule_idx} for patient {pid}")
            print(f"Processing nodule {nodule_idx} for patient {pid}")

            try:
                # Generate a consensus mask for the nodule based on multiple annotations
                mask_3d, cbbox, masks = consensus(nodule, clevel=args.consensus_level)
                logging.debug(f"Nodule {nodule_idx} bounding box: {cbbox}")

                # Resample the mask to match the resampled volume's spacing
                mask_resampled, _ = resample_scan(
                    mask_3d.astype(np.float32),
                    original_spacing,
                    new_spacing=[1.0, 1.0, 1.0]
                )
                mask_resampled = (mask_resampled > 0.5).astype(np.uint8)  # Binarize the mask

                # Extract and resize the nodule to a fixed size for consistent feature extraction
                nodule_vol_resized = extract_and_resize_nodule(
                    vol_normalized,
                    mask_resampled,
                    crop_size_mm=CROP_SIZE_MM
                )

                # Check if the resized nodule is valid (non-empty)
                if np.all(nodule_vol_resized == 0):
                    logging.warning(f"Nodule {nodule_idx} in patient {pid} is empty after resizing. Skipping.")
                    print(f"Nodule {nodule_idx} in patient {pid} is empty after resizing. Skipping.")
                    continue  # Skip to the next nodule

                # Normalize the resized nodule to have zero mean and unit variance
                nodule_vol_normalized = normalize_nodule(nodule_vol_resized)

                # Convert the nodule volume to a PyTorch tensor and add batch and channel dimensions
                volume_tensor = torch.from_numpy(nodule_vol_normalized).unsqueeze(0).unsqueeze(0).float().to(
                    DEVICE
                )  # Shape: (1, 1, 32, 32, 32)

                # Extract CNN features using the pre-trained ResNet18 model
                with torch.no_grad():  # Disable gradient computation for efficiency
                    cnn_features = cnn_model(volume_tensor)
                    cnn_features = cnn_features.cpu().numpy().flatten()  # Convert to NumPy array

                # Calculate the median malignancy score and determine cancer status
                malignancy, is_cancer = calculate_malignancy(nodule)

                # Prepare a metadata dictionary containing patient and nodule information
                metadata_dict = {
                    'patient_id': pid,
                    'nodule_idx': nodule_idx,
                    'malignancy': malignancy,
                    'is_cancer': is_cancer
                }

                # Convert the normalized nodule volume to SimpleITK images for radiomic feature extraction
                sitk_image = sitk.GetImageFromArray(nodule_vol_normalized.astype(np.float32))
                sitk_mask = sitk.GetImageFromArray((nodule_vol_normalized > 0).astype(np.uint8))

                # Set the spacing for the SimpleITK images (SimpleITK expects spacing in (x, y, z))
                sitk_image.SetSpacing(new_spacing[::-1])
                sitk_mask.SetSpacing(new_spacing[::-1])

                # Extract radiomic features using PyRadiomics
                features = extractor.execute(sitk_image, sitk_mask)

                # Prepare a radiomic features dictionary by copying the metadata and adding radiomic features
                radiomic_dict = metadata_dict.copy()
                for key in features.keys():
                    if not key.startswith('diagnostics_'):
                        radiomic_dict[key] = features[key]

                # Append the radiomic features to the radiomic features list
                radiomic_features_list.append(radiomic_dict)

                # Prepare a CNN features dictionary by copying the metadata and adding CNN features
                cnn_dict = metadata_dict.copy()
                for idx, val in enumerate(cnn_features):
                    cnn_dict[f'resnet3d_feature_{idx}'] = val  # Naming convention for CNN features

                # Append the CNN features to the CNN features list
                cnn_features_list.append(cnn_dict)

                logging.info(f"Extracted features for nodule {nodule_idx} of patient {pid}")
                print(f"Extracted features for nodule {nodule_idx} of patient {pid}")

                # Free memory by deleting large objects and invoking garbage collection
                del sitk_image, sitk_mask, nodule_vol_resized, nodule_vol_normalized, features, volume_tensor, cnn_features
                torch.cuda.empty_cache()  # Clear GPU cache if using GPU
                gc.collect()  # Invoke garbage collection to free up memory

            except Exception as e:
                # Log and print any errors encountered while processing a nodule
                logging.error(
                    f"Error processing nodule {nodule_idx} for patient {pid}: {e}",
                    exc_info=True
                )
                print(f"Error processing nodule {nodule_idx} for patient {pid}: {e}")
                continue  # Continue with the next nodule

    except Exception as e:
        # Log and print any errors encountered while processing the scan
        logging.error(f"Error processing scan for patient {pid}: {e}", exc_info=True)
        print(f"Error processing scan for patient {pid}: {e}")

    # Return the lists of radiomic and CNN features for this scan
    return radiomic_features_list, cnn_features_list

# ===============================
#         MAIN FUNCTION
# ===============================

def main():
    """
    Main function to perform 3D radiomic and CNN feature extraction with parallel processing and configurability.
    """
    # Parse command-line arguments to get configuration parameters
    args = parse_arguments()

    # Set up logging based on the provided log file path
    setup_logging(args.log_file)

    # Initialize PyRadiomics feature extractor with selected feature classes
    # Enabling 'firstorder' and 'shape' features; additional feature classes can be enabled as needed
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName('firstorder')  # First-order statistics
    extractor.enableFeatureClassByName('shape')       # Shape features
    extractor.enableFeatureClassByName('glcm')

    # Initialize lists to hold radiomic and CNN feature dictionaries for all scans
    all_radiomic_features: List[dict] = []
    all_cnn_features: List[dict] = []

    # Define and create the output directory if it doesn't exist
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output CSV file paths
    radiomic_csv = output_dir / "meta_info.csv"    # CSV for radiomic features
    cnn_csv = output_dir / "cnn_features.csv"      # CSV for CNN features

    # Query scans from LIDC-IDRI dataset; apply scan limit if specified
    scan_limit = args.scan_limit
    if scan_limit > 0:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).limit(scan_limit).all()
    else:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()

    # Check if any scans were retrieved; if not, log and exit
    if not scans:
        logging.error("No scans found in the database.")
        print("No scans found in the database.")
        return

    logging.info(f"Starting feature extraction for {len(scans)} scans.")
    print(f"Starting feature extraction for {len(scans)} scans.")

    # Use ThreadPoolExecutor for parallel processing of scans
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit extraction tasks for each scan to the executor
        future_to_scan = {
            executor.submit(extract_features_3d, scan, extractor, args): scan
            for scan in scans
        }

        # Iterate through completed futures with a progress bar
        for future in tqdm(as_completed(future_to_scan), total=len(future_to_scan), desc="Processing Scans"):
            scan = future_to_scan[future]  # Retrieve the scan associated with this future
            try:
                # Get the radiomic and CNN features from the future result
                radiomic_features, cnn_features = future.result()
                # Extend the aggregate lists with features from this scan
                all_radiomic_features.extend(radiomic_features)
                all_cnn_features.extend(cnn_features)
                logging.info(f"Successfully processed scan: {scan.patient_id}")
                print(f"Successfully processed scan: {scan.patient_id}")
            except Exception as e:
                # Log and print any errors encountered while processing a scan
                logging.error(f"Error processing scan {scan.patient_id}: {e}", exc_info=True)
                print(f"Error processing scan {scan.patient_id}: {e}")

    # Convert the lists of radiomic and CNN features to pandas DataFrames
    df_radiomic = pd.DataFrame(all_radiomic_features)
    df_cnn = pd.DataFrame(all_cnn_features)

    # Save the DataFrames to separate CSV files
    df_radiomic.to_csv(radiomic_csv, index=False)
    logging.info(f"Radiomic features saved to {radiomic_csv}.")
    print(f"Radiomic features saved to {radiomic_csv}.")

    df_cnn.to_csv(cnn_csv, index=False)
    logging.info(f"CNN features saved to {cnn_csv}.")
    print(f"CNN features saved to {cnn_csv}.")

    # Final log entry indicating completion
    logging.info("Feature extraction completed.")
    print("Feature extraction completed.")

# ===============================
#         SCRIPT ENTRY POINT
# ===============================

if __name__ == '__main__':
    # Import scipy.ndimage here to avoid issues with multiprocessing in some environments
    import scipy.ndimage as ndimage

    # Execute the main function
    main()
