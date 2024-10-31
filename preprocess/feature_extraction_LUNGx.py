#!/usr/bin/env python3
"""
lungx_feature_extraction.py

This script performs 3D radiomic and CNN-based feature extraction on LUNGx CT scans using PyRadiomics and a pre-trained 3D ResNet from MONAI.
It processes each scan, extracts nodules based on provided positions, creates 3D masks, preprocesses the nodules, extracts features,
and computes radiomic and CNN features. The results, along with labels, are saved into two CSV files named 'lungx_meta_info.csv' for radiomic features and 'lungx_cnn_features.csv' for CNN features for further analysis.
"""

# ===============================
#         IMPORTS
# ===============================

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from radiomics import featureextractor

import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from scipy.ndimage import zoom

import threading  # For thread-local storage to handle models in multi-threading

# ===============================
#         CONFIGURATION
# ===============================

# Paths to the LUNGx dataset directories and files
DICOM_DIR = Path(
    '/home/eduardo/PycharmProjects/Lung-Cancer-Classification-using-CT/manifest-cgqtDj7Y2699835271585651107/SPIE-AAPM Lung CT Challenge')  # Replace with your path
EXCEL_FILE = Path('TestSet_NoduleData_PublicRelease_wTruth.xlsx')  # Replace with your path

# Output directory for CSV files
OUTPUT_DIR = Path('LUNGx_Metadata')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output CSV files
RADIOMIC_CSV = OUTPUT_DIR / 'lungx_meta_info.csv'
CNN_CSV = OUTPUT_DIR / 'lungx_cnn_features.csv'

# Logging configuration
LOG_FILE = 'lungx_feature_extraction.log'

# Number of worker threads
NUM_WORKERS = 1  # Adjust based on your system's capabilities

# Device configuration: Use GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Crop size in millimeters
CROP_SIZE_MM = 32

# Thread-local storage for the CNN model
thread_local = threading.local()

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
        filename=log_file,
        filemode='w',  # Overwrite log file each time
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG  # Set to DEBUG level to capture all logs
    )
    logging.info("Logging is set up.")
    print("Logging is set up.")

# ===============================
#         UTILITY FUNCTIONS
# ===============================

def resample_volume(
    volume: np.ndarray,
    original_spacing: List[float],
    new_spacing: List[float] = [1.0, 1.0, 1.0]
) -> np.ndarray:
    """
    Resample the 3D volume to new voxel spacing using trilinear interpolation.

    Parameters:
        volume (np.ndarray): The original 3D volume with shape (z, y, x).
        original_spacing (List[float]): Original voxel spacing in mm for each axis [z, y, x].
        new_spacing (List[float], optional): Desired voxel spacing in mm for each axis [z, y, x].

    Returns:
        np.ndarray: Resampled 3D volume.
    """
    resize_factor = np.array(original_spacing) / np.array(new_spacing)
    logging.debug(f"Resample resize factor: {resize_factor}")
    resampled_volume = zoom(volume, resize_factor, order=1)
    return resampled_volume

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
        return volume - mean
    return (volume - mean) / std

def extract_nodule(
    volume: np.ndarray,
    center_coords: Tuple[int, int, int],
    crop_size_mm: int = 32,
    spacing: List[float] = [1.0, 1.0, 1.0]
) -> np.ndarray:
    """
    Extract a 3D patch centered around the nodule.

    Parameters:
        volume (np.ndarray): The 3D volume with shape (z, y, x).
        center_coords (Tuple[int, int, int]): The (z, y, x) coordinates of the nodule center.
        crop_size_mm (int, optional): Desired size of the crop in mm for each axis. Defaults to 32.
        spacing (List[float], optional): Spacing of the volume in mm for each axis. Defaults to [1.0, 1.0, 1.0].

    Returns:
        np.ndarray: Extracted nodule volume with shape (crop_size, crop_size, crop_size).
    """
    crop_size_voxels = [int(crop_size_mm / s) for s in spacing]
    half_size = [size // 2 for size in crop_size_voxels]

    z, y, x = center_coords
    z_min = max(z - half_size[0], 0)
    z_max = min(z + half_size[0], volume.shape[0])
    y_min = max(y - half_size[1], 0)
    y_max = min(y + half_size[1], volume.shape[1])
    x_min = max(x - half_size[2], 0)
    x_max = min(x + half_size[2], volume.shape[2])

    logging.debug(f"Extracting nodule at coordinates: z={z}, y={y}, x={x}")
    logging.debug(f"Crop size in voxels: {crop_size_voxels}")
    logging.debug(f"Volume shape: {volume.shape}")
    logging.debug(f"Extraction bounds: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]

    # Pad if necessary
    pad_z = crop_size_voxels[0] - patch.shape[0]
    pad_y = crop_size_voxels[1] - patch.shape[1]
    pad_x = crop_size_voxels[2] - patch.shape[2]

    padding = (
        (0, pad_z) if pad_z > 0 else (0, 0),
        (0, pad_y) if pad_y > 0 else (0, 0),
        (0, pad_x) if pad_x > 0 else (0, 0)
    )

    if any(pad > 0 for pad in [pad_z, pad_y, pad_x]):
        patch = np.pad(patch, padding, mode='constant', constant_values=0)

    logging.debug(f"Patch shape after extraction and padding: {patch.shape}")

    return patch

def get_cnn_model() -> nn.Module:
    """
    Initialize the pre-trained ResNet18 model for 3D feature extraction.

    Returns:
        nn.Module: The CNN model.
    """
    if not hasattr(thread_local, 'cnn_model'):
        cnn_model = resnet18(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            num_classes=128,
            shortcut_type='A',
            bias_downsample=True  # Set bias_downsample to True
        ).to(DEVICE)

        cnn_model.eval()
        thread_local.cnn_model = cnn_model

    return thread_local.cnn_model

def create_spherical_mask(shape: Tuple[int, int, int], radius: int) -> np.ndarray:
    """
    Create a spherical mask within a 3D volume.

    Parameters:
        shape (Tuple[int, int, int]): The shape of the volume (z, y, x).
        radius (int): The radius of the sphere.

    Returns:
        np.ndarray: A 3D numpy array representing the spherical mask.
    """
    center = np.array(shape) // 2
    grid = np.ogrid[[slice(0, s) for s in shape]]
    distance = sum((g - c)**2 for g, c in zip(grid, center))
    mask = distance <= radius**2
    return mask.astype(np.uint8)

# ===============================
#         FEATURE EXTRACTION
# ===============================

def extract_features_lungx(nodule_info: dict) -> Tuple[Optional[dict], Optional[dict]]:
    try:
        patient_id = nodule_info['patient_id']
        slice_number = nodule_info['slice_number']
        x_coord = nodule_info['x_coord']
        y_coord = nodule_info['y_coord']
        label = nodule_info['label']
        nodule_id = nodule_info['nodule_id']

        logging.info(f"Processing nodule {nodule_id} for patient {patient_id}")

        # Load the DICOM images for the patient using pydicom
        patient_dir = DICOM_DIR / patient_id

        if not patient_dir.exists():
            logging.warning(f"Patient directory {patient_dir} does not exist. Skipping patient {patient_id}.")
            return None, None

        # Recursively find all files under the patient directory
        all_files = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if not file.startswith('.'):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            logging.warning(f"No files found for patient {patient_id}. Skipping.")
            return None, None

        slices = []
        instance_numbers = []
        for f in all_files:
            try:
                ds = pydicom.dcmread(f)
                image = ds.pixel_array
                slices.append((int(ds.InstanceNumber), image, ds))
                instance_numbers.append(int(ds.InstanceNumber))
            except Exception as e:
                logging.warning(f"File {f} is not a valid DICOM file or cannot be read. Skipping. Error: {e}")
                continue

        if not slices:
            logging.warning(f"No valid DICOM slices found for patient {patient_id}. Skipping.")
            return None, None

        # Sort slices by InstanceNumber
        slices.sort(key=lambda x: x[0])

        # Extract images and stack into volume
        images = [s[1] for s in slices]
        volume = np.stack(images, axis=0)  # Shape: (z, y, x)

        # Get original spacing from the first slice
        ds = slices[0][2]
        spacing_z = float(ds.SliceThickness)
        spacing_x, spacing_y = map(float, ds.PixelSpacing)  # Note the order of x and y
        original_spacing = [spacing_z, spacing_y, spacing_x]  # Adjusted to match (z, y, x)
        logging.info(f"Original spacing for patient {patient_id}: {original_spacing}")

        # Map the nodule's slice number to the index in the volume
        if slice_number not in instance_numbers:
            logging.warning(f"Slice number {slice_number} not found for patient {patient_id}. Skipping nodule {nodule_id}.")
            return None, None
        z_index = instance_numbers.index(slice_number)

        # Adjust coordinates
        y_coord = int(y_coord)
        x_coord = int(x_coord)

        # Resample the volume to isotropic spacing
        volume_resampled = resample_volume(volume, original_spacing, new_spacing=[1.0, 1.0, 1.0])

        # Clamp Hounsfield Units
        volume_clamped = clamp_hounsfield_units(volume_resampled)

        # Normalize intensity
        volume_normalized = normalize_intensity(volume_clamped)

        # Adjust coordinates if resampling was done
        resample_factor = np.array(original_spacing) / np.array([1.0, 1.0, 1.0])
        logging.debug(f"Resample factor: {resample_factor}")

        # Adjust coordinates
        z_coord = int(z_index * resample_factor[0])
        y_coord = int(y_coord * resample_factor[1])
        x_coord = int(x_coord * resample_factor[2])

        logging.debug(f"Adjusted coordinates after resampling: z={z_coord}, y={y_coord}, x={x_coord}")
        logging.debug(f"Resampled volume shape: {volume_normalized.shape}")

        # Check if coordinates are within bounds
        if not (0 <= z_coord < volume_normalized.shape[0] and
                0 <= y_coord < volume_normalized.shape[1] and
                0 <= x_coord < volume_normalized.shape[2]):
            logging.warning(f"Nodule coordinates out of bounds after resampling for patient {patient_id}, nodule {nodule_id}. Skipping.")
            return None, None

        # Extract the nodule patch
        nodule_patch = extract_nodule(
            volume_normalized,
            center_coords=(z_coord, y_coord, x_coord),
            crop_size_mm=CROP_SIZE_MM,
            spacing=[1.0, 1.0, 1.0]
        )

        # Check the nodule_patch content
        logging.debug(f"Nodule patch shape: {nodule_patch.shape}")
        logging.debug(f"Nodule patch min: {nodule_patch.min()}, max: {nodule_patch.max()}, mean: {nodule_patch.mean()}")

        # Create a spherical mask
        mask_array = create_spherical_mask(nodule_patch.shape, radius=10)
        logging.debug(f"Mask sum (number of non-zero voxels): {mask_array.sum()}")

        if np.all(nodule_patch == 0) or mask_array.sum() == 0:
            logging.warning(f"Nodule {nodule_id} in patient {patient_id} is empty after extraction. Skipping.")
            return None, None

        # Prepare for CNN feature extraction
        nodule_tensor = torch.from_numpy(nodule_patch).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        # Extract CNN features
        cnn_model = get_cnn_model()
        with torch.no_grad():
            cnn_features = cnn_model(nodule_tensor)
            cnn_features = cnn_features.cpu().numpy().flatten()
            logging.debug(f"CNN features shape: {cnn_features.shape}")

        # Prepare metadata
        metadata_dict = {
            'patient_id': patient_id,
            'nodule_id': nodule_id,
            'is_cancer': label
        }

        # Prepare CNN features dictionary
        cnn_dict = metadata_dict.copy()
        for idx, val in enumerate(cnn_features):
            cnn_dict[f'resnet3d_feature_{idx}'] = val

        # Prepare for radiomic feature extraction
        sitk_image = sitk.GetImageFromArray(nodule_patch.astype(np.float32))
        sitk_mask = sitk.GetImageFromArray(mask_array)
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        sitk_mask.SetSpacing((1.0, 1.0, 1.0))

        # Extract radiomic features
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape')
        extractor.enableFeatureClassByName('glcm')

        features = extractor.execute(sitk_image, sitk_mask)

        # Prepare radiomic features dictionary
        radiomic_dict = metadata_dict.copy()
        for key in features.keys():
            if not key.startswith('diagnostics_'):
                radiomic_dict[key] = features[key]

        logging.info(f"Successfully processed nodule {nodule_id} for patient {patient_id}")

        return radiomic_dict, cnn_dict

    except Exception as e:
        logging.error(f"Error processing nodule {nodule_id} for patient {patient_id}: {e}", exc_info=True)
        return None, None

# ===============================
#         MAIN FUNCTION
# ===============================

def main():
    """
    Main function to perform 3D radiomic and CNN feature extraction on the LUNGx dataset.
    """
    # Set up logging
    setup_logging(LOG_FILE)

    # Load the Excel file containing nodule information
    nodule_df = pd.read_excel(EXCEL_FILE)

    # Update the required columns to match your Excel file
    required_columns = ['Scan Number', 'Nodule Number', 'Nodule Center x,y Position*', 'Nodule Center Image', 'Final Diagnosis']
    if not all(col in nodule_df.columns for col in required_columns):
        raise ValueError(f"The Excel file must contain the following columns: {required_columns}")

    # Prepare a list of nodule information dictionaries
    nodule_info_list = []

    for _, row in nodule_df.iterrows():
        # Map 'Scan Number' to 'patient_id'
        patient_id = str(row['Scan Number'])

        # Check if 'Nodule Number' is missing
        if pd.isna(row['Nodule Number']):
            logging.warning(f"Nodule Number is missing for patient {patient_id}. Skipping this row.")
            continue  # Skip this row

        # Map 'Nodule Number' to 'nodule_id'
        nodule_id = int(row['Nodule Number'])

        # Check for missing or invalid 'Nodule Center x,y Position*'
        if pd.isna(row['Nodule Center x,y Position*']):
            logging.warning(f"Nodule Center x,y Position* is missing for patient {patient_id}, nodule {nodule_id}. Skipping.")
            continue

        # Split 'Nodule Center x,y Position*' into x and y coordinates
        try:
            x_str, y_str = row['Nodule Center x,y Position*'].split(',')
            x_coord = int(x_str.strip())
            y_coord = int(y_str.strip())
        except ValueError as e:
            logging.warning(f"Invalid position format for patient {patient_id}, nodule {nodule_id}: {e}. Skipping.")
            continue

        # Check if 'Nodule Center Image' is missing
        if pd.isna(row['Nodule Center Image']):
            logging.warning(f"Nodule Center Image is missing for patient {patient_id}, nodule {nodule_id}. Skipping.")
            continue

        # Map 'Nodule Center Image' to 'slice_number'
        slice_number = int(row['Nodule Center Image'])

        # Map 'Final Diagnosis' to 'label'
        if pd.isna(row['Final Diagnosis']):
            logging.warning(f"Final Diagnosis is missing for patient {patient_id}, nodule {nodule_id}. Skipping.")
            continue

        diagnosis = row['Final Diagnosis'].strip().lower()
        if diagnosis == 'benign nodule':
            label = 0
        elif diagnosis in ['primary lung nodule', 'primary lung cancer', 'suspicious malignant nodule']:
            label = 1
        else:
            logging.warning(f"Unknown diagnosis '{diagnosis}' for patient {patient_id}, nodule {nodule_id}. Skipping.")
            continue  # Skip unknown diagnoses

        nodule_info = {
            'patient_id': patient_id,
            'slice_number': slice_number,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'nodule_id': nodule_id,
            'label': label,
        }
        nodule_info_list.append(nodule_info)

    # Initialize lists to hold radiomic and CNN features
    all_radiomic_features: List[dict] = []
    all_cnn_features: List[dict] = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_nodule = {
            executor.submit(extract_features_lungx, nodule_info): nodule_info
            for nodule_info in nodule_info_list
        }

        for future in tqdm(as_completed(future_to_nodule), total=len(future_to_nodule), desc="Processing Nodules"):
            nodule_info = future_to_nodule[future]
            try:
                radiomic_features, cnn_features = future.result()
                if radiomic_features is not None and cnn_features is not None:
                    all_radiomic_features.append(radiomic_features)
                    all_cnn_features.append(cnn_features)
                else:
                    logging.warning(f"Skipped nodule {nodule_info['nodule_id']} for patient {nodule_info['patient_id']}")
            except Exception as e:
                logging.error(f"Error processing nodule {nodule_info['nodule_id']} for patient {nodule_info['patient_id']}: {e}", exc_info=True)

    # Convert the lists to DataFrames
    df_radiomic = pd.DataFrame(all_radiomic_features)
    df_cnn = pd.DataFrame(all_cnn_features)

    # Save to CSV files
    df_radiomic.to_csv(RADIOMIC_CSV, index=False)
    logging.info(f"Radiomic features saved to {RADIOMIC_CSV}.")
    print(f"Radiomic features saved to {RADIOMIC_CSV}.")

    df_cnn.to_csv(CNN_CSV, index=False)
    logging.info(f"CNN features saved to {CNN_CSV}.")
    print(f"CNN features saved to {CNN_CSV}.")

    logging.info("Feature extraction completed.")
    print("Feature extraction completed.")

# ===============================
#         SCRIPT ENTRY POINT
# ===============================

if __name__ == '__main__':
    main()
