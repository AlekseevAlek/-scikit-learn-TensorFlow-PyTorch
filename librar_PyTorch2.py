# Регрессия
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('C:/Users/raveg/PycharmProjects/pythonProject29/dataset2/weather_prediction_dataset.csv')


X = df[['BASEL_cloud_cover', 'BASEL_humidity', 'BASEL_pressure',
        'BASEL_global_radiation', 'BASEL_precipitation', 'BASEL_sunshine',
        'BASEL_temp_min', 'BASEL_temp_max']].shift(1).dropna()  # Previous observations
y = df['BASEL_temp_mean'].shift(-1).dropna()  # Target variable, shifted one row up

X = X.iloc[:-1]
y = y.iloc[1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

input_size = X_train.shape[1]
model = LinearRegressionModel(input_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Reduced learning rate


for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    pred_y = model(X_train_tensor)
    loss = criterion(pred_y, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()


mae_all = mean_absolute_error(y_test, y_pred)
mse_all = mean_squared_error(y_test, y_pred)

print(f'MAE: {mae_all:.4f}, MSE: {mse_all:.4f}')