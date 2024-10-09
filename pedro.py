import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
# Step 1: Import the Excel file into a DataFrame
excel_file = 'has_header_subset.xlsx'
df = pd.read_excel(excel_file)

# Step 2: Find columns with only one unique value
columns_with_one_unique_value = df.columns[df.nunique() == 1]

# Step 3: Create a subset of the DataFrame that excludes these columns
df_subset = df.drop(columns=columns_with_one_unique_value)

# Step 4: Initialize a list to store matching column pairs
matching_column_pairs = []

# Step 5: Loop through each pair of columns in the subset and check if they match element-wise
for colA in df_subset.columns:
    for colB in df_subset.columns:
        if colA != colB:  # Avoid comparing a column with itself
            if (df_subset[colA] == df_subset[colB]).all():  # Check if all values match
                matching_column_pairs.append((colA, colB))

# Step 6: Display the result of matching column pairs
print("Matching column pairs:")
for pair in matching_column_pairs:
    print(pair)
'''

df = pd.read_csv("metadata_3d.csv")
columns_with_one_unique_value = df.columns[df.nunique() == 1]
print(columns_with_one_unique_value)


matching_column_pairs = []

for colA in df.columns:
    for colB in df.columns:
        if colA != colB:  # Avoid comparing a column with itself
            if (df[colA] == df[colB]).all():  # Check if all values match
                matching_column_pairs.append((colA, colB))


for pair in matching_column_pairs:
    print(pair)













'''
# Step 2: Find columns with only one unique value
columns_with_one_unique_value = df.columns[df.nunique() == 1]

# Step 3: Create a subset of the DataFrame that excludes these columns
df_subset = df.drop(columns=columns_with_one_unique_value)


df_subset.drop("patient_id", axis=1, inplace=True)
df_subset.drop("nodule_idx", axis=1, inplace=True)
df_subset.drop("is_cancer", axis=1, inplace=True)

correlation_matrix = df_subset.corr()


#print(df_subset.corr())

#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix[['malignancy']], vmin=-1, vmax=1)

#plt.show()

malignancy_corr = correlation_matrix['malignancy']

strong_corr_with_malignancy = malignancy_corr[(malignancy_corr > 0.65) | (malignancy_corr < -0.65)]

# Step 4: Remove 'malignancy' itself (as it would have a correlation of 1 with itself)
strong_corr_with_malignancy = strong_corr_with_malignancy.drop('malignancy', errors='ignore')

# Display the results
print(strong_corr_with_malignancy)
'''