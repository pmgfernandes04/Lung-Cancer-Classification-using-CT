# Lung Cancer Classification using CT

This project focuses on classifying lung cancer using Computed Tomography (CT) scan data, specifically in DICOM file format. The goal is to develop a machine learning model capable of accurately identifying lung cancer based on the provided medical imaging data.

## Project Workflow

### Data Preprocessing and Feature Extraction

To handle the high-dimensional nature of CT scan data, we extracted both 2D and 3D features using specialized libraries. Key steps in this process included:

1. **Data Extraction**:
   - We used **PyLidc** to process and annotate the lung nodule data, allowing us to identify regions of interest within the scans.
   - **SimpleITK (sitk)** was used to handle DICOM files and facilitate the extraction of 3D volumetric features.
   - Preprocessing also included resizing and normalization to standardize the data for model input.
   - 
2. **Feature Engineering**:
   - **2D Feature Extraction**: Extracted features from 2D cross-sections of the CT images, including texture and shape features.
   - **3D Feature Extraction**: Extracting features for 3D input, was fine-tuned to capture complex volumetric patterns in the data, which are critical for accurate lung cancer classification.
   - **ResNet-18** for 3D feature extraction, already pre-trained to handle medical images.

### Model Training

The model pipeline includes a stacking ensemble of various base models, including:

- Additional machine learning models (e.g., **Random Forest**, **Support Vector Machine**, **XGBoost**) were incorporated into a stacked ensemble, aiming to combine complementary strengths and improve the final predictive performance.

The ensemble was trained on labeled CT data and validated through cross-validation to ensure robustness and generalizability.

### Evaluation on External Dataset (LungX)

We evaluated the model's performance on an external dataset, *LungX*, to test its generalizability. Unfortunately, an oversight during feature extraction led to incomplete normalization, which impaired the model’s ability to accurately classify cases on the new dataset. Therefore, we only report the performance on the primary LIDC dataset.

### Ethical Considerations

In developing a medical diagnostic model, we carefully considered the ethical implications:

- **Patient Privacy**: All data was handled in compliance with privacy laws and guidelines, such as HIPAA, to ensure patient confidentiality and data security.
- **Clinical Safety**: Recognizing that machine learning models can impact patient diagnosis, ensured that the objective of the project is to aid the decision of a professional, but not decide by itself.
- **Among others**

### Conclusion

This project demonstrates the potential of using machine learning for early lung cancer detection from CT scans, emphasizing an effective feature extraction, to create a model that performs well.