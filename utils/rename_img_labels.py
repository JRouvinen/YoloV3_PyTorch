import os

def rename_images(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    count = 1

    # Rename each file in the folder
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            file_extension = file[-4:]
            new_file_name = str(count) + file_extension
            old_file_path = os.path.join(folder_path, file)
            old_file_path = old_file_path.replace('\\','/')
            new_file_path = os.path.join(folder_path, new_file_name)
            new_file_path = new_file_path.replace('\\','/')
            old_label_path = folder_path.replace('images', 'labels')
            old_label_file = file[:-4]
            old_label_file = old_label_file+'.txt'
            old_label_path = os.path.join(old_label_path, old_label_file)
            old_label_path = old_label_path.replace('\\','/')
            new_label_name = str(count) +'.txt'
            new_label_path = os.path.join(folder_path, new_label_name)
            new_label_path = new_label_path.replace('\\','/')
            #rename jpg files
            os.rename(old_file_path, new_file_path)
            #rename txt files
            os.rename(old_label_path, new_label_path)
            count += 1

    print("Images renamed successfully!")

# Provide the folder path containing the images
path_input = input('Give image folder path: ')
folder_path = path_input
folder_path.replace('\\', '/')
# Call the function to rename the images
rename_images(folder_path)