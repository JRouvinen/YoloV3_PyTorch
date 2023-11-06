import glob
import os

folder_path = input('Give image folder path [/data/<folder name>]: ')
current_dir = folder_path+'/images'
split_pct = 20
split_input = input('Give split percentage [default 20%]: ')
if split_input == '':
    split_pct = 20
#split_pct = 20  # 20% validation set
file_train = open(folder_path+"/train.txt", "w")
file_val = open(folder_path+"/val.txt", "w")
counter = 1
index_test = round(100 / split_pct)
# iterate over files in
# that directory
for filename in os.listdir(current_dir):
    f = os.path.join(current_dir, filename)
    f.replace('\\','/')
    # checking if it is a file
    if os.path.isfile(f):
        if counter == index_test:
            counter = 1
            file_val.write(current_dir + "/" + filename + "\n")
        else:
            file_train.write(current_dir + "/" + filename + "\n")
            counter = counter + 1
file_train.close()
file_val.close()

