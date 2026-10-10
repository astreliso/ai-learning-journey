import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithm.logistic_regression import LogisticRegression

data = pd.read_csv('../../../datasets/iris.csv')
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']
X_axis = 'petal_length'
y_axis = 'petal_width'

# 画初始数据集
for iris_type in iris_types:
    plt.scatter(data.loc[data['class']==iris_type,X_axis],   # 选列筛行
                data.loc[data['class']==iris_type,y_axis],
                label=iris_type)
plt.title('Iris dataset')
plt.legend()
plt.show()

# 提取数据
num_examples = data.shape[0]
X_train = data[[X_axis,y_axis]].values.reshape(num_examples,2)
y_train = data['class'].values.reshape(num_examples,1)

# 训练模型并得到结果
max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0

logistic_regression = LogisticRegression(X_train, y_train, polynomial_degree, sinusoid_degree, normalize_data=True)
thetas, cost_histories = logistic_regression.train(max_iterations)
labels = logistic_regression.unique_labels

# 画Cost曲线
plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
plt.title('Cost Function')
plt.legend()
plt.show()

# 预测结果
y_train_predictions = logistic_regression.predict(X_train)
precision = np.sum(y_train_predictions == y_train)/y_train.shape[0]*100
print('precision =', precision)


# 画执行边界进行展示
# 1. 确定网格范围
x_min, x_max = np.min(X_train[:, 0]), np.max(X_train[:, 0])
y_min, y_max = np.min(X_train[:, 1]), np.max(X_train[:, 1])
samples = 150

# 2. 生成网格（标准做法）
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
xx, yy = np.meshgrid(X, Y)          # ★ xx, yy 形状都是 (samples, samples)

# 3. 把网格展平，一次性预测（向量化）
grid_points = np.c_[xx.ravel(), yy.ravel()]        # 形状 (22500, 2)
predictions = logistic_regression.predict(grid_points).ravel()   # 形状 (22500,)

# 4. 把预测结果恢复成网格形状
Z_SETOSA = (predictions == 'SETOSA').reshape(xx.shape).astype(float)
Z_VERSICOLOR = (predictions == 'VERSICOLOR').reshape(xx.shape).astype(float)
Z_VIRGINICA = (predictions == 'VIRGINICA').reshape(xx.shape).astype(float)

# 5. 画真实散点
for iris_type in iris_types:
    plt.scatter(
        X_train[(y_train == iris_type).flatten(), 0],
        X_train[(y_train == iris_type).flatten(), 1],
        label=iris_type
    )

# 6. 画决策边界
plt.contour(X, Y, Z_VERSICOLOR)
plt.legend()
plt.show()
