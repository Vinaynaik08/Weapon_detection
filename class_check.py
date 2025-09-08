import os

# path to your labels folder
labels_dir = "dataset/train/labels"

class_counts = {}

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f.readlines():
                cls = int(line.split()[0])  # first number is class ID
                class_counts[cls] = class_counts.get(cls, 0) + 1

print("Objects per class:", class_counts)