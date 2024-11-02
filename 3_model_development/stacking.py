from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from linear_discriminant import lda_model
from XGBoost import best_model_xgb

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Load the Combined and Cleaned Dataset
data = pd.read_csv('combined_cleaned_data.csv')

# 2. Prepare X and y
# Assuming 'is_cancer' is your target variable
X = data.drop(['is_cancer'], axis=1)
y = data['is_cancer']

# Map target to integer
if y.dtype == 'object' or y.dtype == 'str':
    y = y.map({'False': 0, 'Ambiguous': 1, 'True': 2})

# 3. Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Handle Class Imbalance (if necessary)
# Check class distribution
print("Class distribution in the training set:")
print(y_train.value_counts())

# Apply SMOTE to balance the classes
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# Define the base models (XGBoost and GBDT)
base_models = [
    ('xgb', best_model_xgb),  # Best XGBoost model from GridSearch
    ('lda', lda_model)
]

# Logistic Regression is commonly used for stacking
meta_model = LogisticRegression()

# Define the Stacking Classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,  # Perform 5-fold cross-validation within stacking
    n_jobs=-1
)

# Train the Stacking Model
stacking_model.fit(X_train_res, y_train_res)

# Evaluate the Stacking Model
y_pred_stack = stacking_model.predict(X_test)
y_prob_stack = stacking_model.predict_proba(X_test)

# Stacking Classification Report
print("Stacking Classification Report:")
print(classification_report(y_test, y_pred_stack))

# Stacking Confusion Matrix
cm_stack = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_stack,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['False', 'Ambiguous', 'True'],
    yticklabels=['False', 'Ambiguous', 'True'],
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Stacking Confusion Matrix')
plt.show()

# Stacking ROC-AUC Score
roc_auc_stack = roc_auc_score(y_test, y_prob_stack, multi_class='ovr')
print("Stacking ROC-AUC Score:", roc_auc_stack)
