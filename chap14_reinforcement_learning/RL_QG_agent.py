import tensorflow as tf
import numpy as np
import os

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        self.learning_rate = 0.001
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 0.1  # epsilon-greedy策略
        self.state_size = 64  # 8x8棋盘
        self.action_size = 64  # 64个可能的位置

        self.init_model()

    def init_model(self):
        # 定义自己的 网络
        self.sess = tf.Session()

        # 输入：棋盘状态
        self.state_input = tf.placeholder(tf.float32, [None, self.state_size], name='state_input')

        # Q网络
        with tf.variable_scope('q_network'):
            h1 = tf.layers.dense(self.state_input, 128, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
            self.q_values = tf.layers.dense(h2, self.action_size)

        # 目标Q值和动作
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
        self.action = tf.placeholder(tf.int32, [None], name='action')

        # 计算损失
        action_one_hot = tf.one_hot(self.action, self.action_size)
        q_value_pred = tf.reduce_sum(self.q_values * action_one_hot, axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_q - q_value_pred))

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # 初始化
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def place(self, state, enables):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。

        # 将state转换为网络输入格式
        state_flat = np.array(state).flatten().reshape(1, -1)

        # epsilon-greedy策略
        if np.random.rand() < self.epsilon:
            # 随机选择一个可行动作
            valid_actions = [i for i, enable in enumerate(enables) if enable]
            if valid_actions:
                action = np.random.choice(valid_actions)
            else:
                action = 0
        else:
            # 选择Q值最大的可行动作
            q_vals = self.sess.run(self.q_values, feed_dict={self.state_input: state_flat})[0]

            # 只考虑可行的动作
            valid_actions = [i for i, enable in enumerate(enables) if enable]
            if valid_actions:
                valid_q_vals = [(i, q_vals[i]) for i in valid_actions]
                action = max(valid_q_vals, key=lambda x: x[1])[0]
            else:
                action = 0

        return action

    def train_step(self, state, action, reward, next_state, done):
        """训练一步"""
        state_flat = np.array(state).flatten().reshape(1, -1)
        next_state_flat = np.array(next_state).flatten().reshape(1, -1)

        # 计算目标Q值
        if done:
            target_q_value = reward
        else:
            next_q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_flat})[0]
            target_q_value = reward + self.gamma * np.max(next_q_values)

        # 更新网络
        _, loss = self.sess.run([self.optimizer, self.loss],
                               feed_dict={
                                   self.state_input: state_flat,
                                   self.action: [action],
                                   self.target_q: [target_q_value]
                               })
        return loss

    def save_model(self):  # 保存 模型
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
   .saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):  # 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def set_epsilon(self, epsilon):
        """设置探索率"""
        self.epsilon = epsilon