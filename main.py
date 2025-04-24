# Diamond Price Prediction using Linear Regression
# Передбачення ціни діамантів за допомогою лінійної регресії
# Предсказание цены алмазов с помощью линейной регрессии

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset from online source
# Завантаження датасету з онлайн-джерела
# Загрузка датасета из онлайн-источника
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
df = pd.read_csv(url)

# Select only numeric features for simplicity
# Вибір лише числових ознак для простоти
# Выбор только числовых признаков для упрощения
df = df[["carat", "depth", "x", "y", "z", "price"]]

# Split features and target variable
# Розділення ознак та цільової змінної
# Разделение признаков и целевой переменной
X = df.drop(columns="price")
y = df["price"]

# Split data into training and testing sets
# Поділ даних на тренувальні та тестові множини
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
# Створення та навчання моделі лінійної регресії
# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
# Здійснення передбачень та оцінка моделі
# Сделать предсказания и оценить модель
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize predictions
# Візуалізація передбачень
# Визуализация предсказаний
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Diamond Price Prediction")
plt.grid(True)
plt.show()
