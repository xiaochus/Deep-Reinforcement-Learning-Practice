# -*- coding: utf-8 -*-
import gym


def try_gym():
    # creat Pendulum env.
    env = gym.make('Pendulum-v0')
    # reset game env.
    env.reset()

    # episodes of game
    random_episodes = 0
    # sum of reward of game per episode
    reward_sum = 0

    while random_episodes < 10:
        # show game
        env.render()
        # random choice a action
        # execute the action
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        # print result and reset ganme env if game done.
        if done:
            random_episodes += 1
            print("Reward for this episode was: {}".format(reward_sum))
            reward_sum = 0
            env.reset()

    env.close()


if __name__ == '__main__':
    try_gym()
