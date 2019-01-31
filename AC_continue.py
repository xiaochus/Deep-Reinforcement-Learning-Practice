# -*- coding: utf-8 -*-
import os
import gym
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from DRL import DRL


class AC(DRL):
    """Actor Critic Algorithms with continuous action.
    """
    def __init__(self):
        super(AC, self).__init__()

        self.env = gym.make('Pendulum-v0')
        self.bound = self.env.action_space.high[0]

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        if os.path.exists('model/actor_acc.h5') and os.path.exists('model/critic_acc.h5'):
            self.actor.load_weights('model/actor_acc.h5')
            self.critic.load_weights('model/critic_acc.h5')

        self.gamma = 0.9

    def _build_actor(self):
        """actor model.
        """
        inputs = Input(shape=(3,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        mu = Dense(1, activation='tanh')(x)
        sigma = Dense(1, activation='softplus')(x)

        out = concatenate([mu, sigma], axis=-1)

        model = Model(inputs=inputs, outputs=out)

        return model

    def _build_critic(self):
        """critic model.
        """
        inputs = Input(shape=(3,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def _actor_loss(self, y_true, y_pred):
        """actor loss function.

        Arguments:
            y_true: (action, reward)
            y_pred: action

        Returns:
            loss: reward loss
        """
        mu, sigma = y_pred[:, 0], y_pred[:, 1]
        action_true, td_error = y_true[:, 0], y_true[:, 1]

        mu = self.bound * mu
        sigma = sigma + 0.1
        action_true = K.reshape(action_true, (-1, 1))

        normal_dist = tf.distributions.Normal(mu, sigma)
        log_prob = normal_dist.log_prob(action_true)
        loss = log_prob * K.flatten(td_error)

        return loss

    def discount_reward(self, next_states, reward):
        """Discount reward for critic

        Arguments:
            next_states: next_states
            rewards: reward of last action.
            done: if game done.
        """
        q = self.critic.predict(next_states)[0][0]
        target = reward + self.gamma * q

        return target

    def choice_action(self, x):
        """choice continuous action from normal distributions.

        Arguments:
            x: state

        Returns:
            action: action
        """
        mu, sigma = self.actor.predict(x)[0]
        action = np.random.normal(mu, sigma, 1)[0]
        action = np.clip(action, -self.bound, self.bound)

        return action

    def train(self, episode):
        """training model.

        Arguments:
            episode: ganme episode

        Returns:
            history: training history
        """
        self.actor.compile(loss=self._actor_loss, optimizer=Adam(lr=0.001))
        self.critic.compile(loss='mse', optimizer=Adam(lr=0.01))

        history = {'episode': [], 'Episode_reward': [],
                   'actor_loss': [], 'critic_loss': []}

        for i in range(episode):
            observation = self.env.reset()
            rewards = []
            alosses = []
            closses = []

            while True:
                x = observation.reshape(-1, 3)

                action = self.choice_action(x)
                next_observation, reward, done, _ = self.env.step([action])
                next_observation = next_observation.reshape(-1, 3)

                rewards.append(reward)

                target = self.discount_reward(next_observation, reward)
                y = np.array([target])
                # TD_error = (r + gamma * next_q) - current_q
                td_error = target - self.critic.predict(x)[0][0]
                # loss1 = mse((r + gamma * next_q), current_q)
                loss1 = self.critic.train_on_batch(x, y)

                y = np.array([[action, td_error]])
                loss2 = self.actor.train_on_batch(x, y)

                observation = next_observation[0]

                alosses.append(loss2)
                closses.append(loss1)

                if done:
                    episode_reward = np.sum(rewards)
                    aloss = np.mean(alosses)
                    closs = np.mean(closses)

                    history['episode'].append(i)
                    history['Episode_reward'].append(episode_reward)
                    history['actor_loss'].append(aloss)
                    history['critic_loss'].append(closs)

                    print('Episode: {} | Episode reward: {} | actor_loss: {:.3f} | critic_loss: {:.3f}'.format(i, episode_reward, aloss, closs))

                    break

        self.actor.save_weights('model/actor_acc.h5')
        self.critic.save_weights('model/critic_acc.h5')

        return history

    def play(self):
        """play game with model.
        """
        print('play...')
        observation = self.env.reset()

        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape(-1, 3)
            action = self.choice_action(x)
            observation, reward, done, _ = self.env.step([action])
 
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                random_episodes += 1
                reward_sum = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    model = AC()

    history = model.train(500)
    model.save_history(history, 'ac_continue.csv')

    model.play()
