import os

def change_first_word(folder_path, new_word):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file_name in files:
        # Check if the file is a text file
        if file_name.endswith(".txt"):
            # Read the contents of the file
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Modify the first word in the file
            if lines:
                words = lines[0].split()
                if words:
                    words[0] = new_word
                    lines[0] = " ".join(words)

                    # Write the modified contents back to the file
                    with open(file_path, "w") as file:
                        file.writelines(lines)

# Example usage
folder_path = "C:/Users\Juha/Documents/AI/datasets/test_set_2/validation/Orange/Label"
new_word = "5"
change_first_word(folder_path, new_word)