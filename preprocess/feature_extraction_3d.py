#!/usr/bin/env python3
"""
feature_extraction_3d.py

This script performs 3D radiomic feature extraction on LIDC-IDRI CT scans using PyRadiomics.
It processes each scan, extracts nodules, creates 3D masks, and computes radiomic features.
The results are saved into a CSV file named 'metadata_3d.csv' for further analysis.
"""

import pylidc as pl
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import logging
import gc
from pylidc.utils import consensus


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        filename='feature_extraction_3d.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO  # Set to INFO level
    )
    logging.info("Logging is set up.")


def inspect_scan(scan):
    """Inspect scan details without visualizing slices."""
    pid = scan.patient_id
    logging.info(f"Inspecting scan for patient {pid}")
    print(f"Inspecting Scan for Patient ID: {pid}")

    try:
        vol = scan.to_volume()  # Returns only the volume
        spacing = scan.spacings  # Retrieve spacing separately

        logging.info(f"Volume shape: {vol.shape}, Spacing: {spacing}")

        # Check if volume is predominantly -2048
        unique_values = np.unique(vol)
        logging.info(f"Unique pixel values in volume: {unique_values[:10]}...")

        # Print pixel value statistics
        logging.info(f"Volume pixel values range from {vol.min()} to {vol.max()}")

        return vol, spacing

    except Exception as e:
        logging.error(f"Error inspecting scan {pid}: {e}", exc_info=True)
        print(f"Error inspecting scan {pid}: {e}")
        return None, None


def extract_features_3d(scan, extractor, features_list):
    """
    Extract 3D radiomic features from a single scan.

    Parameters:
        scan (pylidc.Scan): The scan object.
        extractor (radiomics.featureextractor.RadiomicsFeatureExtractor): The PyRadiomics extractor.
        features_list (list): The list to append feature dictionaries to.
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

        # Get nodules
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
                mask_3d, cbbox, masks = consensus(nodule, clevel=0.5)
                logging.debug(f"Nodule {nodule_idx} bounding box: {cbbox}")

                # Crop the volume and mask to the bounding box
                vol_crop = vol[cbbox]
                mask_crop = mask_3d

                # Check if the cropped volume contains valid data
                if np.all(vol_crop == -2048):
                    logging.warning(f"Nodule {nodule_idx} in patient {pid} has an empty cropped volume. Skipping.")
                    print(f"Nodule {nodule_idx} in patient {pid} has an empty cropped volume. Skipping.")
                    continue

                # Convert to SimpleITK images
                sitk_image = sitk.GetImageFromArray(vol_crop)
                sitk_mask = sitk.GetImageFromArray(mask_crop.astype(np.uint8))

                # Set the spacing (SimpleITK expects spacing in (x, y, z))
                sitk_image.SetSpacing(spacing[::-1])
                sitk_mask.SetSpacing(spacing[::-1])

                # Extract features
                features = extractor.execute(sitk_image, sitk_mask)

                # Prepare the features dictionary
                features_dict = {
                    'patient_id': pid,
                    'nodule_idx': nodule_idx
                }

                # Filter out diagnostic features and add the rest
                for key in features.keys():
                    if not key.startswith('diagnostics_'):
                        features_dict[key] = features[key]

                # Append to the list
                features_list.append(features_dict)

                logging.info(f"Extracted features for nodule {nodule_idx} of patient {pid}")
                print(f"Extracted features for nodule {nodule_idx} of patient {pid}")

                # Free memory
                del sitk_image, sitk_mask, vol_crop, mask_crop, features
                gc.collect()

            except Exception as e:
                logging.error(f"Error processing nodule {nodule_idx} for patient {pid}: {e}", exc_info=True)
                print(f"Error processing nodule {nodule_idx} for patient {pid}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error processing scan for patient {pid}: {e}", exc_info=True)
        print(f"Error processing scan for patient {pid}: {e}")


def main():
    """Main function to perform 3D feature extraction."""
    setup_logging()

    # Initialize PyRadiomics feature extractor with default settings
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()  # Enable all feature classes

    # Initialize a list to hold all feature dictionaries
    features_list = []

    # Query all scans from LIDC-IDRI
    scans = pl.query(pl.Scan).limit(10).all()

    if not scans:
        logging.error("No scans found in the database.")
        print("No scans found in the database.")
        return

    # Iterate through each scan and extract features
    for scan in scans:
        extract_features_3d(scan, extractor, features_list)

    # Convert the list of features to a DataFrame
    df = pd.DataFrame(features_list)

    # Save the DataFrame to a CSV file named
    output_csv = 'Metadata/metadata_3d.csv'
    df.to_csv(output_csv, index=False)
    logging.info(f"Feature extraction completed. Results saved to {output_csv}.")
    print(f"Feature extraction completed. Results saved to {output_csv}.")


if __name__ == '__main__':
    main()
