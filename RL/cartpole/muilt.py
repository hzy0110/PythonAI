import gym
env = gym.make('CartPole-v0')
n_episode = 20
for i_episode in range(n_episode):
    observation = env.reset()
    episode_reward = 0
    while True:
        env.render()
        action = env.action_space.sample() # 随机选
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        state = observation
        if done:
            break
    print ('第{}局得分 = {}'.format(i_episode, episode_reward))
env.close()