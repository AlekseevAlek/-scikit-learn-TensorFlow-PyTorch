# Классификация
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Загрузка предварительно обученной модели для извлечения признаков
base_model = VGG16(weights='imagenet', include_top=False)

# Извлечение признаков из изображений
def extract_features(directory, size=(224, 224)):
    features = []
    labels = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = load_img(img_path, target_size=size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature = base_model.predict(img)
            feature = feature.flatten()
            features.append(feature)
            labels.append(subdir)
    return np.array(features), np.array(labels)

X_train, y_train = extract_features("C:/Users/raveg/PycharmProjects/pythonProject29/img/dataset")

# Обучение логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оценка точности
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f'Accuracy (scikit-learn): {accuracy}')