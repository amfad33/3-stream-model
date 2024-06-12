import pandas as pd

# Load the datasets
df1 = pd.read_csv('HVU_test.csv')  # First CSV with YouTube IDs  (put in HVU test)
df2 = pd.read_csv('Youtube8M.csv')  # Second CSV with YouTube IDs    (put in youtube-8m training)

# Assuming the column with YouTube IDs is named 'youtube_id' in both datasets
youtube_ids_df1 = set(df1['youtube_id'])

# Separate the rows with duplicate YouTube IDs
duplicates_df = df2[df2['youtube_id'].isin(youtube_ids_df1)]

# Filter out the rows with duplicate YouTube IDs from the second dataset
filtered_df2 = df2[~df2['youtube_id'].isin(youtube_ids_df1)]

# Save the new second CSV without the duplicate rows
filtered_df2.to_csv('Youtube8M-filtered.csv', index=False)

# Save the duplicate rows to a new CSV
duplicates_df.to_csv('duplicates.csv', index=False)

# This is used to remove the HVU test videos from youtube 8m train dataset in HVU examination pretraining
