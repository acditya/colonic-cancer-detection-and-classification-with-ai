from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import numpy as np
import os

# filename = input('Enter the name of the model: ')
filename = 'ibrahim.keras'
model = load_model(filename)

annotations_path = 'annotations.csv'
annotations = pd.read_csv(annotations_path)
annotations['Image Path'] = 'images/' + annotations['Image Name']

train_data, test_data = train_test_split(annotations, stratify=annotations['Majority Vote Label'], test_size=0.2, random_state=12)

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='Image Path',
    y_col='Majority Vote Label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
)

test_loss, test_acc = model.evaluate(test_generator)
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

miss = 0
for i in range(len(pred)):
    if ground_truth[i] != pred[i]:
        miss += 1
        base_file_name = os.path.basename(test_generator.filenames[i])
        match = annotations.loc[annotations['Image Name'] == base_file_name, 'Number of Annotators who Selected SSA (Out of 7)']
        annotators = match.values[0]
        print(f'True: {ground_truth[i]} ({annotators} out of 7) - Predicted: {pred[i]}')
print(f'Misclassifications: {miss}')