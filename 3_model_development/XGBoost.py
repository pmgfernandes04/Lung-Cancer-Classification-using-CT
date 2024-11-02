import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Import tqdm and tqdm_joblib for progress visualization
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from itertools import product

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the dataset:
    - Encode target variable.
    - Handle non-numeric features.
    - Standardize features.
    - Handle class imbalance with SMOTE.

    Args:
        data (pd.DataFrame): The raw data.

    Returns:
        X_train_res (np.ndarray): Resampled training features.
        X_test (np.ndarray): Test features.
        y_train_res (np.ndarray): Resampled training labels.
        y_test (np.ndarray): Test labels.
    """
    # Encode target variable
    target_mapping = {'Not Cancer': 0, 'Cancer': 1}
    if data['target'].dtype == 'object':
        data['target'] = data['target'].map(target_mapping)
        logging.info("Target variable encoded.")
    else:
        logging.info("Target variable is already numeric.")

    # Separate features and target
    X = data.drop(['target'], axis=1)
    y = data['target']

    # Ensure all features are numeric
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
        logging.info(f"Dropped non-numeric columns: {non_numeric_cols}")
    else:
        logging.info("All features are numeric.")

    # Handle missing values if any
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
        logging.info("Filled missing values with median.")

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Features standardized.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info("Data split into training and test sets.")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the XGBoost model with hyperparameter tuning.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        XGBClassifier: Trained model.
    """
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }

    total_param_combinations = len(list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['learning_rate'],
        param_grid['subsample'],
        param_grid['colsample_bytree']
    )))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=0,
        n_jobs=-1
    )

    with tqdm_joblib(tqdm(desc="Grid Search", total=total_param_combinations * cv.get_n_splits())):
        grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_}")

    best_model_xgb = grid_search.best_estimator_
    return best_model_xgb

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model (XGBClassifier): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Not Cancer', 'Cancer'])
    logging.info("\nClassification Report:\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Cancer', 'Cancer'],
        yticklabels=['Not Cancer', 'Cancer'],
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    logging.info(f"ROC-AUC Score: {roc_auc:.4f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - XGBoost')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load data
    file_path = 'clean_csv.csv'  # Replace with the actual path to your data
    data = load_data(file_path)

    # Preprocess data
    X_train_res, X_test, y_train_res, y_test = preprocess_data(data)

    # Train model
    model = train_model(X_train_res, y_train_res)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
