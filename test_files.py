import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

annotations_path = 'annotations.csv'
annotations = pd.read_csv(annotations_path)
# annotations['Image Path'] = 'images/' + annotations['Image Name']

train_data, test_data = train_test_split(annotations, stratify=annotations['Majority Vote Label'], test_size=0.2, random_state=12)

with open('test_images.txt', 'w') as file:
    file.write('\n'.join(test_data['Image Name'] + ', ' + test_data['Majority Vote Label']))
