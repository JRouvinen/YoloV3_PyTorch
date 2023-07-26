import os

path = input("Give dataset folder path:")
classes_file = path + "/classes.txt"
labels_dir = path + "/labels"
classes = []
annotations = []
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
    f = os.path.join(labels_dir, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename.endswith(".txt"):
        #print(f)
        file = open(f, "r")
        #print(file.read())
        for line in file:
            annotation_number = line.split(" ")[0]
            annotations[int(annotation_number)] += 1
        file.close()

#print(classes)
#print(annotations)

print("Number of annotations in dataset by class:\n")

for i in range(len(classes)):
    print(classes[i] + "-"+str(annotations[i]))
print(f"\nTotal number of annotations: {sum(annotations)}")
