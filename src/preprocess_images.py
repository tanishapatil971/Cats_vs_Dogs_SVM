import os
import cv2
import numpy as np

cat_path = "data/PetImages/Cat"
dog_path = "data/PetImages/Dog"

data = []
labels = []

IMG_SIZE = 64

# Load Cats
for img_name in os.listdir(cat_path)[:1000]:
    try:
        img_path = os.path.join(cat_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img.flatten())
        labels.append(0)

    except:
        pass

# Load Dogs
for img_name in os.listdir(dog_path)[:1000]:
    try:
        img_path = os.path.join(dog_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img.flatten())
        labels.append(1)

    except:
        pass

X = np.array(data)
y = np.array(labels)

print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)
