# python: 2.7
# encoding: utf-8

import numpy as np


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""

        # 请补全此处代码
        self.n_hidden = n_hidden
        self.n_observe = n_observe

        # 初始化权重和偏置
        self.W = np.random.randn(n_observe, n_hidden) * 0.01
        self.b_h = np.zeros(n_hidden)  # 隐藏层偏置
        self.b_v = np.zeros(n_observe)  # 可见层偏置

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def sample_h_given_v(self, v):
        """Sample hidden units given visible units."""
        h_prob = self.sigmoid(np.dot(v, self.W) + self.b_h)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample

    def sample_v_given_h(self, h):
        """Sample visible units given hidden units."""
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.b_v)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample

    def train(self, data, learning_rate=0.1, epochs=10, batch_size=100):
        """Train model using data."""

        # 请补全此处代码
        n_samples = data.shape[0]
        data = data.reshape(n_samples, -1)  # 展平图像

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                v0 = data[batch_indices]

                # Positive phase
                h0_prob, h0_sample = self.sample_h_given_v(v0)

                # Negative phase (Gibbs sampling)
                v1_prob, v1_sample = self.sample_v_given_h(h0_sample)
                h1_prob, h1_sample = self.sample_h_given_v(v1_sample)

                # 计算梯度并更新参数
                positive_grad = np.dot(v0.T, h0_prob)
                negative_grad = np.dot(v1_sample.T, h1_prob)

                self.W += learning_rate * (positive_grad - negative_grad) / batch_size
                self.b_v += learning_rate * np.mean(v0 - v1_sample, axis=0)
                self.b_h += learning_rate * np.mean(h0_prob - h1_prob, axis=0)

            # 计算重构误差
            v_prob, _ = self.sample_v_given_h(h0_sample)
            error = np.mean((v0 - v_prob) ** 2)
            print('Epoch %d, reconstruction error: %.4f' % (epoch + 1, error))

    def sample(self, n_samples=1, n_gibbs_steps=1000):
        """Sample from trained model."""

        # 请补全此处代码
        # 从随机可见层状态开始
        v = np.random.rand(n_samples, self.n_observe)

        # 进行多步Gibbs采样
        for _ in range(n_gibbs_steps):
            h_prob, h_sample = self.sample_h_given_v(v)
            v_prob, v = self.sample_v_given_h(h_sample)

        return v.reshape(n_samples, 28, 28)


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print mnist.shape

    # construct rbm model
    rbm = RBM(2, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    s = rbm.sample()
