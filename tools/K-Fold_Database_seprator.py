import pandas as pd
from sklearn.model_selection import KFold
import os


# Step 1: Combine all CSV files
def combine_csv_files(file_list):
    df_list = []
    for file in file_list:
        df = pd.read_csv(file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


# Step 2: Filter rows where a specific column is False
def filter_dataframe(df, column_name):
    filtered_df = df[df[column_name] != False]
    return filtered_df


# Step 3: Split the data into train and test sets (80% - 20%)
def split_data(df):
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    return train_df, test_df


# Step 4: Create 5-fold cross-validation datasets and save them
def create_and_save_folds(train_df, test_df, output_dir):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fold = 1
    for train_index, val_index in kf.split(train_df):
        fold_train_df = train_df.iloc[train_index]
        fold_val_df = train_df.iloc[val_index]

        fold_train_path = os.path.join(output_dir, f'train_fold_{fold}.csv')
        fold_val_path = os.path.join(output_dir, f'val_fold_{fold}.csv')

        fold_train_df.to_csv(fold_train_path, index=False)
        fold_val_df.to_csv(fold_val_path, index=False)

        fold += 1

    # Save the test set
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)


# Example usage:
file_list = [f"G:\\HVU Downloader\\HVU_Train_part{i}.csv" for i in range(1, 483)]
combined_df = combine_csv_files(file_list)
filtered_df = filter_dataframe(combined_df,
                               'file')
train_df, test_df = split_data(filtered_df)
create_and_save_folds(train_df, test_df,
                      'G:\\HVU Downloader\\')
