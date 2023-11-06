import os


def delete_files_with_long_names(directory):
    files_removed = 0
    for filename in os.listdir(directory):
        if len(filename) > 2:
            os.remove(os.path.join(directory, filename))
            files_removed += 1
    print(f'{files_removed} files removed')


# Call the function and set the path to your directory
delete_files_with_long_names("C:/Users/juha-matti.rouvinen/PycharmProjects/YoloV3_PyTorch/data/test_set_v3_updated/images")
