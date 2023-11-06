import os


def change_first_word(folder_path, new_value):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file_name in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read all lines from the file
                lines = file.readlines()

            # Open the file in write mode
            with open(file_path, 'w') as file:
                # Iterate through each line
                for line in lines:
                    # Split the line into words
                    words = line.split()

                    # Check if there are words in the line
                    if words:
                        # Replace the first word with the new value
                        words[0] = new_value

                    # Write the modified line to the file
                    file.write(' '.join(words) + '\n')


# Example usage
folder_path = '/path/to/your/folder'
new_value = 'NEW_WORD'
change_first_word(folder_path, new_value)