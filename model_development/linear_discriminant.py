import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# primeira parte igual a do passos

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

print("Class distribution after resampling:")
print(y_train_res.value_counts())

# 6. Initialize the Linear Discriminant Analysis model
lda_model = LinearDiscriminantAnalysis()

# 7. Train the LDA model
lda_model.fit(X_train_res, y_train_res)

# 8. Make Predictions on the Test Set
y_pred = lda_model.predict(X_test)
y_prob = lda_model.predict_proba(X_test)

# 9. Evaluate the LDA Model

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
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
plt.title('LDA Confusion Matrix')
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
class_names = ['False', 'Ambiguous', 'True']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc_dict[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class (LDA)')
plt.legend(loc='lower right')
plt.show()
