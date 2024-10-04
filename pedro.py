import pandas as pd

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