# Регрессия
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

df = pd.read_csv('C:/Users/raveg/PycharmProjects/pythonProject29/dataset2/weather_prediction_dataset.csv')

# Входные данные (X) и целевая переменная (y)
X_all = df[['BASEL_cloud_cover', 'BASEL_humidity', 'BASEL_pressure',
            'BASEL_global_radiation', 'BASEL_precipitation', 'BASEL_sunshine',
            'BASEL_temp_min', 'BASEL_temp_max']].shift(1).dropna()  # Предшествующие наблюдения
y_all = df['BASEL_temp_mean'].shift(-1).dropna()  # Целевая переменная, смещённая на одну строку вверх

# Разделение данных на обучающую и тестовую выборки
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Создание модели линейной регрессии с использованием Keras
model_all = Sequential([
    Dense(1, input_dim=X_train_all.shape[1], activation='linear')
])

# Компиляция модели
model_all.compile(optimizer='adam', loss='mse')

# Обучение модели
model_all.fit(X_train_all, y_train_all, epochs=100, batch_size=10, verbose=1)

# Предсказания и оценка модели
y_pred_all = model_all.predict(X_test_all).flatten()
mae_all = mean_absolute_error(y_test_all, y_pred_all)
mse_all = mean_squared_error(y_test_all, y_pred_all)

print(mae_all, mse_all)