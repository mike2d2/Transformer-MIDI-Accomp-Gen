# Load the list of genre labels from the file
with open('preprocessing/msd_tagtraum_cd1.cls', 'r') as f:
    genre_labels = [line.strip() for line in f]

# Display the list of genre labels
print("List of possible genre labels:")
for label in genre_labels:
    print(label)