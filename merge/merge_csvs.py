import pandas as pd

# Load the CSV files
df1 = pd.read_csv('cnn_features.csv')
df2 = pd.read_csv('meta_info_3d.csv')

# Find common columns
common_columns = [col for col in df1.columns if col in df2.columns]

# For common columns, keep only one if they are identical in both DataFrames
for col in common_columns:
    if df1[col].equals(df2[col]):
        df2 = df2.drop(columns=[col])  # Drop from df2 if identical in df1 and df2

# Now concatenate the DataFrames on index, without duplicating identical columns
merged_df = pd.concat([df1, df2], axis=1)

# Export the merged DataFrame to a new CSV file
merged_df.to_csv('3d_and_deep.csv', index=False)

# Verify column count
print(len(merged_df.columns))
