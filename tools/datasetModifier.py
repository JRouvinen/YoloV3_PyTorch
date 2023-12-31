import os

path = input("Give dataset to modify folder path:")
classes_file = path + "/classes.txt"
new_classes_file = path + "/classes_new.txt"
labels_dir = path + "/labels"
mapping = {"0":"0",
            "1":"2",
            "2":"3",
            "3":"1",
            "4":"4",
            "5":"6",
            "6":"7",
            "7":"5",
            "8":"8",
            "9":"10",
            "10":"11",
            "11":"9",
            "12":"12",
            "13":"14",
            "14":"15",
            "15":"13",
            "16":"20",
            "17":"16",
            "18":"18",
            "19":"19",
            "20":"17",
            "21":"18",
            "22":"12",
            "23":"13",
            "24":"13",
            "25":"13",
            "26":"14",
            "27":"15",
            "28":"20",
            "29":"16",
            "30":"16",
            "31":"16",
            "32":"17",
            "33":"17",
            "34":"17",
            "35":"18",
            "36":"19",
            }

classes = []
annotations = []
files_modified = 0
#read classes file and store it in a list
f = open(classes_file, "r")
#print(f.read())
for line in f:
    line_to_list = line.replace("\n", "")
    classes.append(line_to_list)
    annotations.append(0)
f.close()
#print(classes)

#read label files and store number of labels it in a list
for filename in os.listdir(labels_dir):
    file_data_to_write = ""
    f = os.path.join(labels_dir, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename.endswith(".txt"):
        try:
            #print(f)
            file = open(f, "r")
            #print(file.read())
            for line in file:
                #print(file)
                line_data = line.split(" ")
                #print(line_data)
                annotation_number = line_data[0]
                #print(annotation_number)
                new_annotation_number = mapping[annotation_number]
                line_data[0] = new_annotation_number
                file_data_to_write += " ".join(line_data)
            file.close()
            file_to_overwrite = open(f, "w")
            print(f"Rewriting file annotation: {f}")
            file_to_overwrite.write(file_data_to_write)
            file_to_overwrite.close()
            files_modified += 1
        except KeyError:
            print(f"Error file handling file: {f}")

#print(classes)
#print(annotations)

print(f"Number of files modified:{files_modified}\n")

