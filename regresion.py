"""step 1 imports"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""step 2 creating dataset"""

data = {
    'amount of steps': [2500, 12100, 3231, 12123, 7434, 14654, 1754, 12312, 13234, 11231, 9342, 12313, 12133,
                        5313, 13211],
    'sleep(hours)': [4, 8, 5, 8, 4, 9, 2, 6, 7, 8, 6, 8, 7, 2, 7],
    'water(liters)': [1, 3.7, 1.7, 3.5, 2, 3, 1, 2.7, 3.5, 3.7, 3, 3.9, 3, 1, 3],
    'burned calories': [250, 1210, 323, 1212, 743, 1465, 175, 1231, 3234, 1131, 934, 1213, 1233,
                        531, 1311]
}

table = pd.DataFrame(data)

print(table.head())

"""step 3 separate all data on test and train"""

X = table[['amount of steps', 'sleep(hours)', 'water(liters)']]
y = table['burned calories']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""step 4 creating and training linear regression model"""
model = LinearRegression()

model.fit(x_train, y_train)

"""step 5 printing all information about line and loss"""
print(f"model coeficient:{model.coef_}")
print(f"model bias:{model.intercept_}")

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
print(f"MSE:{mse}")

"""step 6 creating a plot"""
plt.scatter(y_test, y_predict)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Real calories burned')
plt.ylabel('Predicted calories')
plt.title('Real and predicted burned calories')
plt.show()
