import os

# Define the directory to search (current directory)
directory = '.'

# Traverse through all files in the directory and subdirectories
for foldername, subfolders, filenames in os.walk(directory):
    for filename in filenames:
        if filename.endswith('.py'):
            filepath = os.path.join(foldername, filename)
            
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # Replace occurrences of 'int' with 'int'
            new_content = content.replace('int', 'int')

            # Write the updated content back to the file
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(new_content)

            print(f"Updated {filepath}")
