import os
import random

folder_path = input('Give image folder path [/data/<folder name>]: ')
current_dir = folder_path+'/images'
split_pct = [70,20,10]
split_input = input('Give split percentage (x,y,z) [default train 70%, validate 20%, test 10%]: ')
if split_input != '':
    split_pct = split_input.split(',')
#split_pct = 20  # 20% validation set
file_train = open(folder_path+"/train.txt", "w")
file_val = open(folder_path+"/validate.txt", "w")
file_test = open(folder_path+"/test.txt", "w")
counter = 1
#index_test = round(100 / split_pct)
# Count files in the defined folder
print('Counting number of label files in folder...')
number_of_files = 0
for filename in os.listdir(current_dir):
    f = os.path.join(current_dir, filename)
    f.replace('\\', '/')
    # checking if it is a file
    file_added = False
    if os.path.isfile(f):
        if filename.endswith('.jpg'):
            number_of_files += 1
print(f'{number_of_files} label files found.')
# iterate over files in
print('Splitting into train / validate / test files')
files_to_train = round(number_of_files*(int(split_pct[0])/100))
files_to_validate = round(number_of_files*(int(split_pct[1])/100))
files_to_test = round(number_of_files*(int(split_pct[2])/100))
validate_files_inx = random.sample(range(0, number_of_files+1), files_to_validate)
test_files_inx = random.sample(range(0, number_of_files+1), files_to_test)
set_val = set(validate_files_inx)
set_test = set(test_files_inx)
if set_val == set_test:
    test_files_inx = random.sample(range(0, number_of_files+1), files_to_test)
files_in_train = 0
files_in_validate = 0
files_in_test = 0
last_added_index = None
index = 0
# that directory
for filename in os.listdir(current_dir):
    f = os.path.join(current_dir, filename)
    f.replace('\\','/')
    # checking if it is a file
    file_added = False
    if os.path.isfile(f):
        if index in validate_files_inx:
            file_val.write(current_dir + "/" + filename + "\n")
        if index in test_files_inx:
            file_test.write(current_dir + "/" + filename + "\n")
        if index not in validate_files_inx and index not in test_files_inx:
            file_train.write(current_dir + "/" + filename + "\n")
    index += 1
file_train.close()
file_val.close()
file_test.close()
print(f'Files created into {folder_path}')


