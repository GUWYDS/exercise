# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        # 请补全此处代码
        self.W = None
        self.b = None

    def train(self, data_train, learning_rate=0.001, epochs=1000, C=1.0):
        """
        训练模型。
        """
        # 请补全此处代码
        X = data_train[:, :2]
        y = data_train[:, 2]
        # 将标签转换为 -1 和 1
        y = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.W = np.zeros(n_features)
        self.b = 0

        # 梯度下降训练
        for epoch in range(epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.W) + self.b) >= 1
                if condition:
                    # 正确分类，只更新权重正则化项
                    self.W -= learning_rate * (2 * 1/epochs * self.W)
                else:
                    # 错误分类，更新权重和偏置
                    self.W -= learning_rate * (2 * 1/epochs * self.W - np.dot(x_i, y[idx]))
                    self.b -= learning_rate * (-y[idx])

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        linear_output = np.dot(x, self.W) + self.b
        # 将预测结果转换为 0 和 1
        return np.where(linear_output >= 0, 1, 0)


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
