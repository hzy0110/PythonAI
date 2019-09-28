# import gym
# import time
#
# env = gym.make('CartPole-v0')  # 获得游戏环境
# observation = env.reset()  # 复位游戏环境,新一局游戏开始
# print('新一局游戏 初始观测 = {}'.format(observation))
# for t in range(200):
#     env.render()
#     action = env.action_space.sample()  # 随机选择动作
#     print('{}: 动作 = {}'.format(t, action))
#     observation, reward, done, info = env.step(action)  # 执行行为
#     print('{}: 观测 = {}, 本步得分 = {}, 结束指示 = {}, 其他信息 = {}'.format(
#         t, observation, reward, done, info))
#     if done:
#         break
#     # time.sleep(1)#可加可不加，有的话就可以看到图
#
# env.close()
import numpy as np
# # print(0%500)
#
# a = np.random.randint(10,size=(3,4))
# print(a)
# print("---------------")
# # dataSet=np.array(a)
#
# print(a[2, :])
# print(a[:, 2])
# print(a[:, 3])
# print("---------------")
# print(np.arange(100, dtype=np.int32))
import random

# print(np.random.uniform(-2, 2))
memory = np.zeros((500, 3 * 2 + 2))
# memory = np.zeros((500, 4 * 2 + 2))
# transition = np.hstack(([1,2,3], [[1], 5], [4,5,6]))
# transition = np.hstack(([-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289]))
# transition = np.array([-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289])
# print(transition.shape)
# transition = np.array([[-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289]])
# print(transition.shape)
# [-0.04939213 -0.02850458  0.03744223 -0.03893088  0.
#  0.50412655  -0.04996222 -0.22414289  0.03666361  0.26532635]
# transition = [-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289, 0.03666361,
#               0.26532635]

# transition = [-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289]
#
# memory[0, :] = transition
# print(memory)

a = np.random.randint(100, size=(30, 5))
print(a)
# b = a[:, -3:]
# print(b)
# c = a[:, :3]
# print(c)
# d = a[:, 3]
# print(d)

# e= 0.9 * np.max(a, axis=1)
# print(e)
#
f = 1 + 0.9 * np.max(a, axis=1)
# (30,)
# print("f", f.shape)
# q_target <class 'numpy.ndarray'>
# eval_act_index <class 'numpy.ndarray'> [15  9 14 26 18  7 18 16 10 11 25 15 28 26  9  7 12 13 14 29 27 29 23  5
#  21 29 13 14 20 19  1  8  2 29 26 29 15  3 16 14  2 17  5 15 20 13 10  1
#  12  5 15 10 23  7  8 26  1  5 20  2 23 16 25  1]
g = np.arange(30, dtype=np.int32)
# print(g)
# print("g", type(g))
# h = a[:, 3].astype(int)
# print("h.shape", h.shape,h)
h1 = np.random.randint(4, size=(30,))
print("h1", h1)
i = a.copy()
print("ig", i[g, h1])
# i1 = i[g, h1]
# print(i[3, 4])

# transition = [-0.04939213, -0.02850458, 0.03744223, -0.03893088, 0., 0.50412655, -0.04996222, -0.22414289]

# import gym
#
# env = gym.make('Pendulum-v0')  # 定义使用 gym 库中的那一个环境
# env = env.unwrapped  # 不做这个会有很多限制
# # from gym import envs
# # print(envs.registry.all())
# print("env.action_space", env.action_space)  # 查看这个环境中可用的 action 有多少个
# print("env.observation_space", env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
# print("env.observation_space.high", env.observation_space.high)  # 查看 observation 最高取值
# print("env.observation_space.low", env.observation_space.low)  # 查看 observation 最低取值
# print("env.action_space", env.action_space)
# print("nv.observation_space.shape[0]", env.observation_space.shape[0])


# Pendulum-v0
# env.action_space Box(1,)
# env.observation_space Box(3,)
# env.observation_space.high [1. 1. 8.]
# env.observation_space.low [-1. -1. -8.]
# env.action_space Box(1,)
# nv.observation_space.shape[0] 3

# 'CartPole-v1'
# env.action_space Discrete(2)
# env.observation_space Box(4,)
# env.observation_space.high [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
# env.observation_space.low [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
# env.action_space Discrete(2)
# nv.observation_space.shape[0] 4
# reward = 5
# reward /= 10
# print(reward)