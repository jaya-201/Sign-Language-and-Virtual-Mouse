import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------- SETTINGS ---------
IMG_SIZE = 64
DATASET_DIR = 'dataset'
categories = sorted(os.listdir(DATASET_DIR))  # ['A', 'B', 'C', ...]

# --------- LOAD DATA ---------
X = []
y = []

for label_index, label in enumerate(categories):
    folder = os.path.join(DATASET_DIR, label)
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_index)

X = np.array(X, dtype='float32') / 255.0
y = to_categorical(y, num_classes=len(categories))

# --------- SPLIT DATA ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- DATA AUGMENTATION ---------
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# --------- MODEL ---------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------- TRAIN ---------
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10,
                    validation_data=(X_test, y_test))

# --------- SAVE ---------
model.save('sign_model.h5')
np.save('categories.npy', categories)

print("Model trained and saved successfully.")