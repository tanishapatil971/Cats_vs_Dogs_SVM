import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Dataset Paths

cat_path = "data/PetImages/Cat"
dog_path = "data/PetImages/Dog"


# Parameters

IMG_SIZE = 64
MAX_IMAGES = 1000

data = []
labels = []


# Load Cat Images

for img_name in os.listdir(cat_path)[:MAX_IMAGES]:
    try:
        img_path = os.path.join(cat_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img / 255.0

        data.append(img.flatten())
        labels.append(0)  # Cat

    except:
        pass


# Load Dog Images

for img_name in os.listdir(dog_path)[:MAX_IMAGES]:
    try:
        img_path = os.path.join(dog_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img / 255.0

        data.append(img.flatten())
        labels.append(1)  # Dog

    except:
        pass


# Convert to NumPy Arrays

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded:", X.shape)


# Feature Scaling

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# Train SVM Model

print("\nTraining SVM...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("Training Complete!")


# Predictions

y_pred = model.predict(X_test)


# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# Sample Predictions

print("\n===== SAMPLE PREDICTIONS =====")

for i in range(10):
    actual = "Cat" if y_test[i] == 0 else "Dog"
    predicted = "Cat" if y_pred[i] == 0 else "Dog"

    print(
        f"Image {i+1}: Actual = {actual}, Predicted = {predicted}"
    )