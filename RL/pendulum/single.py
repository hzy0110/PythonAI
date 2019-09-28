import gym

env = gym.make('Pendulum-v0')
# n_episode = 20
# for i_episode in range(n_episode):
observation = env.reset()
episode_reward = 0
# while True:
# env.render()
action = env.action_space.sample()  # 随机选
print("env.action_space.sample()", env.action_space.sample())
observation, reward, done, _ = env.step(action)
# Pendulum
# observation [-0.85450557 -0.51944223 -0.30856382]
# reward -6.659042474543858
# done False
# _ {}
print("observation", observation)
print("reward", reward)
print("done", done)
print("_", _)

episode_reward += reward
state = observation
# if done:
#     break
# print ('第{}局得分 = {}'.format(i_episode, episode_reward))
env.close()
