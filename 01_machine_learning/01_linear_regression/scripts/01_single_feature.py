import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithm.linear_regression import LinearRegression

data = pd.read_csv('../../../datasets/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

X_train = train_data[[input_param_name]].values         # numpy二维数组
y_train = train_data[[output_param_name]].values

X_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

'''
测试
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(type(X_train))
print(X_train.min())
'''

num_iterations = 500
learning_rate = 0.01

linearregression = LinearRegression(X_train, y_train)
(theta, cost_history) = linearregression.train(learning_rate, num_iterations)

print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.title('GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

predictions_num = 100
x_predictions = np.linspace(X_train.min(), X_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linearregression.predict(x_predictions)

plt.scatter(X_train,y_train,label='Train data',color='orange')
plt.scatter(X_test,y_test,label='Test data',color='blue')
plt.plot(x_predictions, y_predictions,label='LinearRegression',color='red')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Linear Regression')
plt.legend()
plt.show()

