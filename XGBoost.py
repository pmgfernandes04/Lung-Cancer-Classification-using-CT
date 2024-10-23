# gradient_boosting_classifier.py

# 1. Import Necessary Libraries
import pandas as pd
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
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Import tqdm and tqdm_joblib for progress visualization
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# 2. Load the Combined and Cleaned Dataset
data = pd.read_csv('combined_cleaned_data.csv')  # Replace with your actual filename

# 3. Prepare X and y
# Assuming 'is_cancer' is your target variable
X = data.drop(['is_cancer'], axis=1)
y = data['is_cancer']

# If 'is_cancer' is not already encoded, encode target variable
if y.dtype == 'object' or y.dtype == 'str':
    y = y.map({'False': 0, 'Ambiguous': 1, 'True': 2})

# 4. Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Handle Class Imbalance (if necessary)
# Check class distribution
print("Class distribution in the training set:")
print(y_train.value_counts())

# Apply SMOTE to balance the classes
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Class distribution after resampling:")
print(y_train_res.value_counts())

# 7. Set Up the XGBoost Classifier
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)

# 8. Hyperparameter Tuning (Optional)
# Define a grid of hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
}

# Calculate total number of parameter combinations
from itertools import product

total_param_combinations = len(list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['subsample'],
    param_grid['colsample_bytree']
)))

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',  # Use appropriate scoring metric
    cv=cv,
    verbose=0,  # Suppress verbose output to keep tqdm output clean
    n_jobs=-1
)

# Wrap the grid search with tqdm_joblib for progress bar
with tqdm_joblib(tqdm(desc="Grid Search", total=total_param_combinations * cv.get_n_splits())):
    grid_search.fit(X_train_res, y_train_res)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Use the best estimator
best_model = grid_search.best_estimator_

# 9. Train the Model
best_model.fit(X_train_res, y_train_res)

# 10. Evaluate the Model
# Predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['False', 'Ambiguous', 'True'],
    yticklabels=['False', 'Ambiguous', 'True'],
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)

# ROC Curves

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc_dict = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['blue', 'green', 'red']
class_names = ['False', 'Ambiguous', 'True']  # Adjust based on your encoding
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc_dict[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc='lower right')
plt.show()
