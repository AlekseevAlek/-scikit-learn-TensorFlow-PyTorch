# Регрессия
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Загрузка данных из файла CSV
df = pd.read_csv('C:/Users/raveg/PycharmProjects/pythonProject29/dataset2/weather_prediction_dataset.csv')

# Просмотр первых 5 строк датасета
print(df.head())
print(df.info())

data2 = pd.read_csv('C:/Users/raveg/PycharmProjects/pythonProject29/dataset2/weather_prediction_bbq_labels.csv')
print(data2.head())

# Модель 1: Использование только средней температуры для предсказания средней температуры

# Входные данные (X) и целевая переменная (y)
X_temp = df[['BASEL_temp_mean']].shift(1).dropna()  # Предшествующие наблюдения
y_temp = df['BASEL_temp_mean'][1:]  # Целевая переменная, смещённая на одну строку вверх

# Разделение данных на обучающую и тестовую выборки
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model_temp = LinearRegression()
model_temp.fit(X_train_temp, y_train_temp)

# Предсказания и оценка модели
y_pred_temp = model_temp.predict(X_test_temp)
mae_temp = mean_absolute_error(y_test_temp, y_pred_temp)
mse_temp = mean_squared_error(y_test_temp, y_pred_temp)

print (mae_temp, mse_temp)

# Модель 2: Использование всех погодных параметров для предсказания средней температуры

# Входные данные (X) и целевая переменная (y)
X_all = df[['BASEL_cloud_cover', 'BASEL_humidity', 'BASEL_pressure',
            'BASEL_global_radiation', 'BASEL_precipitation', 'BASEL_sunshine',
            'BASEL_temp_min', 'BASEL_temp_max']].shift(1).dropna()  # Предшествующие наблюдения
y_all = df['BASEL_temp_mean'][1:]  # Целевая переменная, смещённая на одну строку вверх

# Разделение данных на обучающую и тестовую выборки
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)

# Предсказания и оценка модели
y_pred_all = model_all.predict(X_test_all)
mae_all = mean_absolute_error(y_test_all, y_pred_all)
mse_all = mean_squared_error(y_test_all, y_pred_all)

print(mae_all, mse_all)

import matplotlib.pyplot as plt

# Визуализация реальной и предсказанной температуры для обеих моделей

# Модель 1: Предсказание средней температуры на основе прошлых температур
plt.figure(figsize=(14, 6))

# График для первой модели
plt.subplot(1, 2, 1)
plt.plot(y_test_temp.values, label='Реальная температура', color='blue')
plt.plot(y_pred_temp, label='Предсказанная температура (Модель 1)', color='orange', linestyle='--')
plt.title('Модель 1: Средняя температура (только прошлые температуры)')
plt.xlabel('Наблюдения')
plt.ylabel('Температура (°C)')
plt.legend()

# График для второй модели
plt.subplot(1, 2, 2)
plt.plot(y_test_all.values, label='Реальная температура', color='blue')
plt.plot(y_pred_all, label='Предсказанная температура (Модель 2)', color='green', linestyle='--')
plt.title('Модель 2: Средняя температура (все параметры)')
plt.xlabel('Наблюдения')
plt.ylabel('Температура (°C)')
plt.legend()

plt.tight_layout()
plt.show()