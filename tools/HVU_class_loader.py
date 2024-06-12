import csv

# Step 1: Read the CSV file
input_file = 'G:\\HVU Downloader\\HVU_Train_V1.0.csv'
tags_column = 'Tags'
output_file = 'G:\\HVU Downloader\\HVU_Classes.csv'

# Initialize a set to store unique tags
unique_tags = set()

# Read the input CSV and extract tags
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        tags = row[tags_column].split('|')
        unique_tags.update(tags)

# Step 2: Assign an ID to each unique tag
tags_with_ids = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}

# Step 3: Write the results to a new CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Tag', 'ID'])
    for tag, idx in tags_with_ids.items():
        writer.writerow([tag, idx])

print(f"Unique tags with IDs have been written to {output_file}")
