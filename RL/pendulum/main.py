import gym
import numpy as np
import math
import time
from pendulum.RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym.make('Pendulum-v0')  # 定义使用 gym 库中的那一个环境
# 动作空间： (-2, 2) 力矩
# 观察空间： ([-1,-1,-8],[1,1,8])
# cos 和 sin 值 和 最大角速度
env = env.unwrapped  # 不做这个会有很多限制
ACTION_SPACE = 30
MEMORY_SIZE = 4000
# print("env.action_space", env.action_space)  # 查看这个环境中可用的 action 有多少个
# print("env.observation_space", env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
# print("env.observation_space.high", env.observation_space.high)  # 查看 observation 最高取值
# print("env.observation_space.low", env.observation_space.low)  # 查看 observation 最低取值
# # print("env.action_space.n", env.action_space.n)
# print("nv.observation_space.shape[0]", env.observation_space.shape[0])

# 定义使用 DQN 的算法
DQN = DeepQNetwork(n_actions=ACTION_SPACE,  # 1
                  n_features=env.observation_space.shape[0],  # 3
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE, batch_size=32,
                  e_greedy_increment=0.001, )


def train(RL):
    print("开始")
    acc_r = [0]
    total_steps = 0  # 记录步数
    observation = env.reset()

    # learn_steps = 1000  # 设置开始学习的步数
    # 达到最大步数，就认为完成目标，设置为 done
    # max_episode_steps = 3000
    # theta_threshold_radians = 12 * 2 * math.pi / 360
    # observation = env.reset()
    # for i1 in range(100):
    #     env.render()
    #     observation_, reward, done, info = env.step(1)
    #     print(observation_, reward, done, info)


    # for i_episode in range(100):
    # manual_done = False
    # while_steps = 0  # 内循环步数
    # 获取回合 i_episode 第一个 observation

    # print("observation", observation)
    # ep_r = 0
    while True:

        if total_steps - MEMORY_SIZE > 9000: env.render()
        if total_steps % 1000 == 0: print(total_steps)

        # env.render()  # 刷新环境

        action = RL.choose_action(observation)  # 选行为
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        # action = env.action_space.sample() # 随机选
        # print("action2", type(action), action)
        # observation_, reward, done, _ = env.step(action)  # 获取下一个state
        observation_, reward, done, info = env.step(np.array([f_action]))  # 获取下一个state

        cos, sin, thetadot = observation_  # 细分开, 为了修改原配的 reward
        # print("cos, sin", cos, sin)
        # print("------------------------------")
        # print(" reward:", reward, " action:", action, " f_action:", f_action, " cos, sin:", cos + sin, " env.state:", env.state[0] + env.state[1], "thetadot", thetadot)

        # angle_normalize = (((cos + np.pi) % (2 * np.pi)) - np.pi)

        # reward = angle_normalize**2 + .1*thetadot**2 + .001*(action**2)

        # x, x_dot, theta, theta_dot = observation_
        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (theta_threshold_radians - abs(sin)) / theta_threshold_radians - 0.5
        # print("rrr", r2, reward)
        # reward = -r2
        # reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
        acc_r.append(reward + acc_r[-1])
        # 保存这一组记忆
        reward /= 10
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            # print("RL.learn()")
            RL.learn()  # 学习

        if total_steps-MEMORY_SIZE > 15000:
            break
        observation = observation_
        total_steps += 1
        # while_steps += 1
        # time.sleep(0.1)

        # print("observation_", observation_)
        # print("reward", reward)
        # print("done", done)
        # print("_", _)

        # print("action", action)
        # print("env.g", env.g)
        # print("env.dt", env.dt)
        # print("env.last_u", env.last_u)
        # print("env.pole_transform", env.pole_transform)
        # print("env.reward_range", env.reward_range)
        # print("env.spec", env.spec)
        # print("env.state", env.state)
        # print("env.unwrapped", env.unwrapped)

    return RL.cost_his, acc_r


c_natural, r_natural = train(DQN)
# print("开始画图")
# plt.figure(1)
# plt.plot(np.array(c_natural), c='r', label='natural')
# # plt.plot(np.array(c_dueling), c='b', label='dueling')
# plt.legend(loc='best')
# plt.ylabel('cost')
# plt.xlabel('training steps')
# plt.grid()
#
# plt.figure(2)
# plt.plot(np.array(r_natural), c='r', label='natural')
# # plt.plot(np.array(r_dueling), c='b', label='dueling')
# plt.legend(loc='best')
# plt.ylabel('accumulated reward')
# plt.xlabel('training steps')
# plt.grid()
#
# plt.show()

# 最后输出 cost 曲线
# RL.plot_cost()
