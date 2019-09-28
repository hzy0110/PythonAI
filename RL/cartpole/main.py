# encoding:UTF-8

# Cart Pole Environment
import gym

env = gym.make('CartPole-v0')

# 搭建 DQN
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, env.action_space.n)
)

import random


def act(model, state, epsilon):
    if random.random() > epsilon:  # 选最大的
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value = model.forward(state)
        action = q_value.max(1)[1].item()
    else:  # 随便选
        action = random.randrange(env.action_space.n)
    return action


# 训练
# epsilon值不断下降
import math


def calc_epsilon(t, epsilon_start=1.0,
                 epsilon_final=0.01, epsilon_decay=500):
    epsilon = epsilon_final + (epsilon_start - epsilon_final) \
              * math.exp(-1. * t / epsilon_decay)
    return epsilon


# 最近历史缓存
import numpy as np
from collections import deque

batch_size = 32


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip( \
            *random.sample(self.buffer, batch_size))
        concat_state = np.concatenate(state)
        concat_next_state = np.concatenate(next_state)
        return concat_state, action, reward, concat_next_state, done

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBuffer(1000)

import torch.optim

optimizer = torch.optim.Adam(model.parameters())

gamma = 0.99

episode_rewards = []  # 各局得分,用来判断训练是否完成
t = 0  # 训练步数,用于计算epsilon

while True:

    # 开始新的一局
    state = env.reset()
    episode_reward = 0

    while True:
        epsilon = calc_epsilon(t)
        action = act(model, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) > batch_size:
            # 计算时间差分误差
            sample_state, sample_action, sample_reward, sample_next_state, \
            sample_done = replay_buffer.sample(batch_size)

            sample_state = torch.tensor(sample_state, dtype=torch.float32)
            sample_action = torch.tensor(sample_action, dtype=torch.int64)
            sample_reward = torch.tensor(sample_reward, dtype=torch.float32)
            sample_next_state = torch.tensor(sample_next_state,
                                             dtype=torch.float32)
            sample_done = torch.tensor(sample_done, dtype=torch.float32)

            next_qs = model(sample_next_state)
            next_q, _ = next_qs.max(1)
            expected_q = sample_reward + gamma * next_q * (1 - sample_done)

            qs = model(sample_state)
            q = qs.gather(1, sample_action.unsqueeze(1)).squeeze(1)

            td_error = expected_q - q

            # 计算 MSE 损失
            loss = td_error.pow(2).mean()

            # 根据损失改进网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t += 1

        if done:  # 本局结束
            i_episode = len(episode_rewards)
            print('第{}局收益 = {}'.format(i_episode, episode_reward))
            episode_rewards.append(episode_reward)
            break

    if len(episode_rewards) > 20 and np.mean(episode_rewards[-20:]) > 195:
        break  # 训练结束

# 使用 （固定 ϵ 的值为0）
n_episode = 20
for i_episode in range(n_episode):
    observation = env.reset()
    episode_reward = 0
    while True:
        # env.render()
        action = act(model, observation, 0)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        state = observation
        if done:
            break
    print('第{}局得分 = {}'.format(i_episode, episode_reward))