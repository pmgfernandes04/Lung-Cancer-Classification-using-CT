# simple script to merge 3d and cnn csvs into a desided csv file

import pandas as pd

# read the input, assuming the datasets are on the script_generated_csvs folder
df1_name = str(input(f"Name of the first dataset, including the .csv extension: \n"))
df2_name = str(input(f"Name of the second dataset, including the .csv extension: \n"))

dir_name = "./script_generated_csvs/"
df1_name = dir_name + df1_name
df2_name = dir_name + df2_name

df1 = pd.read_csv(df1_name)
df2 = pd.read_csv(df2_name)

# Find common columns
common_columns = [col for col in df1.columns if col in df2.columns]

# For common columns, keep only one if they are identical in both DataFrames
for col in common_columns:
    if df1[col].equals(df2[col]):
        df2 = df2.drop(columns=[col])  # Drop from df2 if identical in df1 and df2

# Now concatenate the DataFrames on index, without duplicating identical columns
merged_df = pd.concat([df1, df2], axis=1)

# Export the merged DataFrame to a new CSV file
merged_csv = str(input(f"Final merged dataset name, including the .csv extension: \n"))
merged_df.to_csv(merged_csv, index=False)

# Verify column count
print(len(merged_df.columns))
