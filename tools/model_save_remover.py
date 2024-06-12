import os


def remove_old_parts(base_path, current_id):
    """
    Removes all files with the pattern base_path+file_name+f"_part{i}.pth"
    where i < current_id - 9.

    Parameters:
    base_path (str): The base path to the directory containing the files.
    file_name (str): The base name of the files to be removed.
    current_id (int): The current ID to determine which files to remove.

    Returns:
    List[str]: List of removed files.
    """

    # List to store the names of the files that are removed
    removed_files = []

    # Calculate the threshold ID
    threshold_id = current_id - 9

    # Iterate over the range of IDs less than the threshold ID
    for i in range(threshold_id):
        # Construct the file path
        file_path = base_path + f"_ep{i}.pth"

        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)
            # Append the removed file to the list
            removed_files.append(file_path)

    return removed_files
