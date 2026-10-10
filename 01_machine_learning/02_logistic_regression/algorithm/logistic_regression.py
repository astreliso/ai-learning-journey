import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid

class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data = True ):  #如果不要归一化直接改这里
        """
        1. 对数据进行预处理操作
        2. 先得到所有的特征个数
        3. 初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_features_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_features_labels, num_features))    # θ = (类别数行, 特征+1列)

    def train(self, max_iterations=1000):
        cost_histories = []
        num_features = self.data.shape[1]
        # 遍历的同时取出索引和值
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_labels = (self.labels == unique_label).astype(float)
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data,current_labels, current_initial_theta,max_iterations)
            self.theta[label_index:label_index+1] = current_theta.T
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations):
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标
            lambda current_theta: LogisticRegression.cost_function(data,labels, current_theta.reshape(num_features, 1)),
            # 初始化权重参数
            x0 = current_initial_theta.flatten(),
            # 选择优化策略（共轭梯度下降）
            method = 'CG',
            # 梯度函数 (要求一维）
            jac =  lambda current_theta: LogisticRegression.gradient_step(data,labels, current_theta.reshape(num_features, 1)),
            # 记录结果
            callback = lambda current_theta: cost_history.append(LogisticRegression.cost_function(data,labels, current_theta.reshape(num_features, 1))),
            # 迭代次数
            options = {'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Logistic regression failed'+result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, theta):
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data,theta)
        label_diff = predictions - labels
        gradients = (1/num_examples)*np.dot(data.T,label_diff)
        return gradients.flatten()  # jac要求的梯度是一维(n,)

    @staticmethod
    def cost_function(data, labels, theta):
        """
        计算交叉熵损失函数
        """
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        y_is_set_cost = np.dot(labels[labels == 1].T,np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1-labels[labels == 0].T,np.log(1-predictions[labels == 0]))
        cost = (-1/num_examples)*(y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data ,
                                              features_mean=self.features_mean,
                                              features_deviation=self.features_deviation)[0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        # 取概率最大的类索引（一行一行看，找每行中最大的）
        max_prob_index = np.argmax(prob, axis=1)
        # 索引映射为类名
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((num_examples, 1))


