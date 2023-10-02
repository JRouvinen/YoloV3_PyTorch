import os


def delete_files_with_long_names(directory):
    files_removed = 0
    for filename in os.listdir(directory):
        if len(filename) > 10:
            os.remove(os.path.join(directory, filename))
            files_removed += 1
    print(f'{files_removed} files removed')


# Call the function and set the path to your directory
delete_files_with_long_names("C:/Users/Kimiwaha/AI/datasets/ITDM_15488_12345_interf/labels")
