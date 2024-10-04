#!/usr/bin/env python3
"""
main.py

This script processes multiple LIDC-IDRI scans using pylidc, extracting nodule information and saving relevant data.
It has been optimized to reduce memory usage and prevent Out of Memory (OOM) issues.
"""

import gc
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from statistics import median_high

import numpy as np
import pandas as pd
import pylidc as pl
import SimpleITK as sitk

from radiomics import featureextractor
import albumentations as A

from utils import is_dir_path, segment_lung
from pylidc.utils import consensus

# Configure Logging
logging.basicConfig(
    filename='prepare_dataset.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set to INFO for detailed logs
)

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

# Get Directory settings
DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset', 'MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset', 'IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
META_PATH = is_dir_path(parser.get('prepare_dataset', 'META_PATH'))

# Hyperparameter settings for prepare_dataset function
mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')

# Hyperparameter settings for pylidc
confidence_level = parser.getfloat('pylidc', 'confidence_level')
padding_size = parser.getint('pylidc', 'padding_size')


class MakeDataSet:
    """
    A class to prepare and process a single LIDC-IDRI scan for machine learning applications.
    """

    def __init__(self, scan, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE,
                 CLEAN_DIR_MASK, META_DIR, mask_threshold, padding_size, confidence_level=0.5):
        self.scan = scan
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding_size, padding_size)] * 3

        # Initialize Radiomics Feature Extractor
        radiomics_params = {'enableAllFeatures': True}
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_params)
        self.extractor.enableFeatureClassByName('shape2D')

        # Data Augmentation Pipeline
        self.augmentation_pipeline = self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),  # Apply horizontal flips with a probability of 50%
            A.ShiftScaleRotate(
                shift_limit=0.02,  # Shift images by up to 2% of height/width
                scale_limit=0.05,  # Scale images by up to ±5%
                rotate_limit=10,  # Rotate images within ±10 degrees
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,  # Adjust brightness by up to ±10%
                contrast_limit=0.05,  # Adjust contrast by up to ±10%
                p=0.2
            ),
            A.ElasticTransform(
                alpha=0.5,
                sigma=30,
                alpha_affine=None,
                p=0.1
            ),  # Elastic deformation
            A.GaussNoise(
                var_limit=(5.0, 10.0),
                p=0.2
            )  # Add Gaussian noise to simulate noisy scans
        ])

        # Initialize metadata DataFrame with basic columns
        self.meta_columns = ['patient_id', 'nodule_no', 'slice_no', 'original_image', 'mask_image',
                             'malignancy', 'is_cancer', 'is_clean']
        self.meta = pd.DataFrame(columns=self.meta_columns)
        logging.info("Initialized MakeDataSet class.")

    def calculate_malignancy(self, nodule_annotations):
        """
        Calculate the malignancy of a nodule based on radiologist annotations.

        Parameters:
            nodule_annotations (list): List of Annotation objects.

        Returns:
            tuple: (malignancy score, cancer label)
        """
        list_of_malignancy = [annotation.malignancy for annotation in nodule_annotations]
        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self, meta_dict):
        """
        Saves the information of a nodule to the metadata DataFrame.

        Parameters:
            meta_dict (dict): Dictionary containing metadata elements.
        """
        # Convert meta_dict to a DataFrame
        meta_df = pd.DataFrame([meta_dict])
        # Concatenate with the existing self.meta DataFrame
        self.meta = pd.concat([self.meta, meta_df], ignore_index=True)

    def assess_data_quality(self):
        """
        Assess the quality of the scan data.

        Returns:
            bool: True if scan passes quality checks, False otherwise.
        """
        # Check if 'slice_zvals' attribute exists and has slices
        if hasattr(self.scan, 'slice_zvals'):
            num_slices = len(self.scan.slice_zvals)
            if num_slices == 0:
                logging.warning(f"Scan {self.scan.id} has zero slices.")
                return False
        else:
            logging.warning(f"Scan {self.scan.id} does not have 'slice_zvals'.")
            return False

        # Additional data quality checks
        # Check for consistent slice thickness
        try:
            dicom_images = self.scan.load_all_dicom_images()
            slice_thicknesses = [float(img.SliceThickness) for img in dicom_images]
            if len(set(slice_thicknesses)) > 1:
                logging.warning(f"Scan {self.scan.id} has inconsistent slice thickness.")
                return False
        except AttributeError:
            logging.warning(f"Scan {self.scan.id} missing SliceThickness information.")
            return False

        # Check for missing or corrupt slices
        expected_slices = len(dicom_images)
        if num_slices != expected_slices:
            logging.warning(f"Scan {self.scan.id} is missing slices.")
            return False

        return True

    def normalize_hounsfield_units(self, image, hu_min=-1000, hu_max=400):
        """
        Normalizes Hounsfield Units (HU) to a range between 0 and 1.

        Parameters:
            image (numpy.ndarray): The image slice.
            hu_min (int): Minimum HU value to clip.
            hu_max (int): Maximum HU value to clip.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Clip to the specified HU range
        image = np.clip(image, hu_min, hu_max)
        # Normalize to 0-1
        image = (image - hu_min) / (hu_max - hu_min)
        return image

    def extract_radiomics_features(self, image, mask):
        """
        Extract radiomic features from an image and its corresponding mask.

        Parameters:
            image (numpy.ndarray): The image slice.
            mask (numpy.ndarray): The mask slice.

        Returns:
            dict: Radiomic features.
        """
        # Ensure image is in the correct format
        image = image.astype(np.float32)
        mask = mask.astype(np.int16)  # Masks should be integer type

        # Convert to SimpleITK images
        image_sitk = sitk.GetImageFromArray(image)
        mask_sitk = sitk.GetImageFromArray(mask)

        # Ensure that the mask is cast to an integer type
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkInt16)

        # Extract features
        features = self.extractor.execute(image_sitk, mask_sitk)
        # Prefix radiomics features
        radiomics_features = {f'radiomics_{k}': v for k, v in features.items() if not k.startswith('diagnostics')}

        # Free memory
        del image_sitk, mask_sitk
        gc.collect()

        return radiomics_features

    def augment_data(self, image, mask):
        """
        Applies data augmentation to the image and mask.

        Parameters:
            image (numpy.ndarray): The image slice.
            mask (numpy.ndarray): The mask slice.

        Returns:
            tuple: Augmented image and mask.
        """
        augmented = self.augmentation_pipeline(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def process_scan(self):
        """
        Processes a single patient's scan, extracting features, saving images, masks, and metadata.
        """
        # Naming prefix for files
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Create necessary directories
        for path in [self.img_path, self.mask_path, self.clean_path_img, self.clean_path_mask, self.meta_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {path}")

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)

        pid = self.scan.patient_id
        logging.info(f"Processing patient: {pid}")
        print(f"Processing patient: {pid}")

        # Assess data quality
        if not self.assess_data_quality():
            logging.info(f"Scan {self.scan.id} failed quality assessment. Skipping.")
            print(f"Scan {self.scan.id} failed quality assessment. Skipping.")
            return

        try:
            nodules_annotation = self.scan.cluster_annotations()

            num_slices = len(self.scan.slice_zvals)
            logging.info(
                f"Patient ID: {pid}, Number of Slices: {num_slices}, Number of Annotated Nodules: {len(nodules_annotation)}")
            print(
                f"Patient ID: {pid}, Number of Slices: {num_slices}, Number of Annotated Nodules: {len(nodules_annotation)}")

            if len(nodules_annotation) > 0:
                # Patients with nodules
                patient_image_dir = IMAGE_DIR / pid
                patient_mask_dir = MASK_DIR / pid
                patient_image_dir.mkdir(parents=True, exist_ok=True)
                patient_mask_dir.mkdir(parents=True, exist_ok=True)

                for nodule_idx, nodule in enumerate(nodules_annotation):
                    try:
                        mask_3d, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                        zmin, zmax = cbbox[2].start, cbbox[2].stop
                        slices = self.scan.load_all_dicom_images(verbose=False)[zmin:zmax]

                        # Process each slice
                        for idx, img in enumerate(slices):
                            image_slice = img.pixel_array
                            image_slice = self.normalize_hounsfield_units(image_slice)
                            mask_slice = mask_3d[:, :, idx]
                            if np.sum(mask_slice) <= self.mask_threshold:
                                continue

                            # Segment Lung region
                            lung_segmented_np_array = segment_lung(image_slice)
                            lung_segmented_np_array = lung_segmented_np_array.astype(np.float32)
                            mask_slice = mask_slice.astype(np.uint8)

                            #Original Images being trnasformed in npy
                            nodule_name = f"{pid[-4:]}_NI{prefix[nodule_idx]}_slice{prefix[idx]}.npy"
                            mask_name = f"{pid[-4:]}_MA{prefix[nodule_idx]}_slice{prefix[idx]}.npy"

                            # Extract Radiomic Features from the ORIGINAL images
                            radiomics_features = self.extract_radiomics_features(lung_segmented_np_array, mask_slice)

                            malignancy, cancer_label = self.calculate_malignancy(nodule)

                            meta_dict = {
                                'patient_id': pid[-4:],
                                'nodule_no': nodule_idx,
                                'slice_no': prefix[idx],
                                'original_image': nodule_name,
                                'mask_image': mask_name,
                                'malignancy': malignancy,
                                'is_cancer': cancer_label,
                                'is_clean': False,
                            }
                            meta_dict.update(radiomics_features)

                            self.save_meta(meta_dict)
                            print(f"Added metadata for nodule {nodule_idx}, slice {idx} in scan {pid}")

                            # Save images and masks
                            np.save(patient_image_dir / nodule_name, lung_segmented_np_array.astype(np.float32))
                            np.save(patient_mask_dir / mask_name, mask_slice.astype(np.uint8))

                            logging.info(f"Processed nodule {nodule_idx} slice {idx} for scan {self.scan.id}")
                            print(f"Processed nodule {nodule_idx} slice {idx} for scan {pid}")

                            #Applying data augmentation for training a Machine Learning Model
                            augmented_image, augmented_mask = self.augment_data(lung_segmented_np_array, mask_slice)
                            np.save(patient_image_dir / f"augmented_{nodule_name}", augmented_image.astype(np.float32))
                            np.save(patient_mask_dir / f"augmented_{mask_name}", augmented_mask.astype(np.uint8))

                            # Release memory
                            gc.collect()

                    except Exception as e:
                        logging.error(f"Error processing nodule {nodule_idx} in scan {pid}: {type(e).__name__}: {e}",
                                      exc_info=True)
                        continue

            # Save metadata to CSV
            self.finalize_dataset()

        except Exception as e:
            logging.error(f"Error processing scan {pid}: {type(e).__name__}: {e}", exc_info=True)

    def finalize_dataset(self):
        """
        Finalizes the dataset preparation by saving metadata.
        """
        logging.info("Saving metadata.")
        print("Saving metadata.")  # Immediate feedback
        # Save metadata to CSV
        meta_file = Path(self.meta_path) / 'meta_info.csv'
        if meta_file.exists():
            # If metadata file exists, append without headers
            self.meta.to_csv(meta_file, mode='a', header=False, index=False)
        else:
            # If metadata file does not exist, save with headers
            self.meta.to_csv(meta_file, index=False)
        logging.info("Metadata saved to meta_info.csv.")


def main():
    # Process scans for the first 10 patients
    # Query the first 10 scans from pylidc
    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).limit(10).all()

    if not scans:
        print("No scans found.")
        return

    for scan in scans:
        patient_id = scan.patient_id
        print(f"Processing scan for patient {patient_id}")

        # Initialize MakeDataSet class for one patient
        dataset_preparer = MakeDataSet(
            scan,
            IMAGE_DIR,
            MASK_DIR,
            CLEAN_DIR_IMAGE,
            CLEAN_DIR_MASK,
            META_PATH,
            mask_threshold,
            padding_size,
            confidence_level
        )

        # Process the dataset for one patient
        try:
            dataset_preparer.process_scan()
        except Exception as e:
            logging.error(f"Error processing patient {patient_id}: {type(e).__name__}: {e}", exc_info=True)


if __name__ == '__main__':
    main()