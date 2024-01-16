# COSC 330 Project 
# Fall 2023

# Team Members:
# Aditya Chatterjee   100061369
# Ayah Miqdady        100061370
# Muhammad Firdousi   100061194
# Tuba Musba          100061362

# a. Library Imports

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50V2
from keras.models import Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint

# b. Reading Dataset into Dataframe

annotations_path = 'annotations.csv'
print(annotations_path)
annotations = pd.read_csv(annotations_path)
annotations['Image Path'] = 'images/' + annotations['Image Name']

# c. Dataset Splitting (Training, Testing, Validation)

train_data, test_data = train_test_split(annotations, stratify=annotations['Majority Vote Label'], test_size=0.2, random_state=12)
train_data, val_data = train_test_split(train_data, stratify=train_data['Majority Vote Label'], test_size=0.2, random_state=12)

# d. Data Augmentation and creating Generators

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True
)

datagen_val_test = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_data,
    x_col='Image Path',
    y_col='Majority Vote Label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
)

val_generator = datagen_val_test.flow_from_dataframe(
    dataframe=val_data,
    x_col='Image Path',
    y_col='Majority Vote Label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
)

test_generator = datagen_val_test.flow_from_dataframe(
    dataframe=test_data,
    x_col='Image Path',
    y_col='Majority Vote Label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
)

# e. Model Creation

base_model = ResNet50V2(
    weights='imagenet',
    include_top=False
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'), 
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[BinaryAccuracy()]
)

# f. Model Training

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.0001
)

best_model = ModelCheckpoint('ibrahim.h5', save_best_only=True)

epochs = 25

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks = [early_stopping, best_model],
)

# g. Evaluation and Confusion Matrix

best_model = load_model('ibrahim.h5')
best_model.save('ibrahim.keras')

test_loss, test_acc = best_model.evaluate(test_generator)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

ground_truth = test_generator.classes

pred = model.predict(test_generator)
pred = np.round(pred)

cm = confusion_matrix(ground_truth, pred)
tn, fp, fn, tp = cm.ravel()
print(f'True Negatives: {tn}')
print(f'False Positives: {fp}')
print(f'False Negatives: {fn}')
print(f'True Positives: {tp}')