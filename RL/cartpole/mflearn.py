import gym
from cartpole.RL_brain import DeepQNetwork

env = gym.make('CartPole-v1')  # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 不做这个会有很多限制

print("env.action_space", env.action_space)  # 查看这个环境中可用的 action 有多少个
print("env.observation_space", env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
print("env.observation_space.high", env.observation_space.high)  # 查看 observation 最高取值
print("env.observation_space.low", env.observation_space.low)  # 查看 observation 最低取值
print("env.action_space.n", env.action_space.n)
print("nv.observation_space.shape[0]", env.observation_space.shape[0])

# 定义使用 DQN 的算法
RL = DeepQNetwork(n_actions=env.action_space.n,  # 2
                  n_features=env.observation_space.shape[0],  # 4
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0008, )

total_steps = 0  # 记录步数

# observation = env.reset()
# for i1 in range(100):
#     env.render()
#     observation_, reward, done, info = env.step(1)
#     print(observation_, reward, done, info)


for i_episode in range(100):

    # 获取回合 i_episode 第一个 observation
    observation = env.reset()
    # print("observation", observation)
    ep_r = 0
    while True:
        env.render()  # 刷新环境

        action = RL.choose_action(observation)  # 选行为
        # print("action", type(action), action)

        observation_, reward, done, info = env.step(action)  # 获取下一个state
        # print("observation_", observation_)
        # print("reward", reward)
        # print("done", done)
        # print("info", info)

        x, x_dot, theta, theta_dot = observation_  # 细分开, 为了修改原配的 reward
        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

        # x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
        print("reward", reward)
        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            # print("RL.learn()")
            RL.learn()  # 学习

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

#
# 最后输出 cost 曲线
# RL.plot_cost()
