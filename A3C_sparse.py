# -*- coding: utf-8 -*-
import os
import gym
import time
import threading

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


# global variables for threading
step = 0
history = {'episode': [], 'Episode_reward': []}
lock = threading.Lock()


class A3C:
    """A3C Algorithms with sparse action.
    """
    def __init__(self):
        self.gamma = 0.95
        self.actor_lr = 0.001
        self.critic_lr = 0.01

        self._build_model()
        self.optimizer = self._build_optimizer()

        # handle error
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self):
        """actor model.
        """
        inputs = Input(shape=(4,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def _build_critic(self):
        """critic model.
        """
        inputs = Input(shape=(4,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def _build_model(self):
        """build model for multi threading training.
        """
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Pre-compile for threading
        self.actor._make_predict_function()
        self.critic._make_predict_function()

    def _build_optimizer(self):
        """build optimizer and loss method.

        Returns:
            [actor optimizer, critic optimizer].
        """
        # actor optimizer
        actions = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))
        action_pred = self.actor.output

        entropy = K.sum(action_pred * K.log(action_pred + 1e-10), axis=1)
        closs = K.binary_crossentropy(actions, action_pred)
        actor_loss = K.mean(closs * K.flatten(advantages)) - 0.01 * entropy

        actor_optimizer = Adam(lr=self.actor_lr)
        actor_updates = actor_optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        actor_train = K.function([self.actor.input, actions, advantages], [], updates=actor_updates)

        # critic optimizer
        discounted_reward = K.placeholder(shape=(None, 1))
        value = self.critic.output

        critic_loss = K.mean(K.square(discounted_reward - value))

        critic_optimizer = Adam(lr=self.critic_lr)
        critic_updates = critic_optimizer.get_updates(self.critic.trainable_weights, [], critic_loss)
        critic_train = K.function([self.critic.input, discounted_reward], [], updates=critic_updates)

        return [actor_train, critic_train]

    def train(self, episode, n_thread, update_iter):
        """training A3C.

        Arguments:
            episode: total training episode.
            n_thread: number of thread.
            update_iter: update iter.
        """
        # Multi threading training.
        threads = [Agent(i, self.actor, self.critic, self.optimizer, self.gamma, episode, update_iter) for i in range(n_thread)]

        for t in threads:
            t.start()
            time.sleep(1)

        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")

        self.save()

    def load(self):
        """Load model weights.
        """
        if os.path.exists('model/actor_a3cs.h5') and os.path.exists('model/critic_a3cs.h5'):
            self.actor.load_weights('model/actor_a3cs.h5')
            self.critic.load_weights('model/critic_a3cs.h5')

    def save(self):
        """Save model weights.
        """
        self.actor.save_weights('model/actor_a3cs.h5')
        self.critic.save_weights('model/critic_a3cs.h5')


class Agent(threading.Thread):
    """Multi threading training agent.
    """
    def __init__(self, index, actor, critic, optimizer, gamma, episode, update_iter):
        threading.Thread.__init__(self)

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.episode = episode
        self.update_iter = update_iter

        self.env = gym.make('CartPole-v0')

    def run(self):
        """training model.
        """
        global history
        global step

        while step < self.episode:
            observation = self.env.reset()

            states = []
            actions = []
            rewards = []

            while True:
                x = observation.reshape(-1, 4)
                states.append(x)

                # choice action with prob.
                prob = self.actor.predict(x)[0][0]
                action = np.random.choice(np.array(range(2)), p=[1 - prob, prob])
                actions.append(action)

                next_observation, reward, done, _ = self.env.step(action)
                next_observation = next_observation.reshape(-1, 4)
                rewards.append(reward)

                observation = next_observation[0]

                if ((step + 1) % self.update_iter == 0) or done:
                    lock.acquire()
                    try:
                        self.train_episode(states, actions, rewards, next_observation, done)

                        if done:
                            episode_reward = sum(rewards)
                            history['episode'].append(step)
                            history['Episode_reward'].append(episode_reward)

                            print('Thread: {} | Episode: {} | Episode reward: {}'.format(self.index, step, episode_reward))

                            step += 1
                    finally:
                        lock.release()

                if done:
                    break

    def discount_reward(self, rewards, next_state, done):
        """Discount reward

        Arguments:
            rewards: rewards in a episode.
            next_states: next state of current game step.
            done: if epsiode done.

        Returns:
            discount_reward: n-step discount rewards.
        """
        # compute the discounted reward backwards through time.
        discount_rewards = np.zeros_like(rewards, dtype=np.float32)

        if done:
            cumulative = 0.
        else:            
            cumulative = self.critic.predict(next_state)[0][0]

        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discount_rewards[i] = cumulative

        return discount_rewards

    def train_episode(self, states, actions, rewards, next_observation, done):
        """training algorithm in an epsiode.
        """
        states = np.concatenate(states, axis=0)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards)

        # Q_values
        values = self.critic.predict(states)
        # discounted rewards
        discounted_rewards = self.discount_reward(rewards, next_observation, done)
        discounted_rewards = discounted_rewards.reshape(-1, 1)
        # advantages
        advantages = discounted_rewards - values

        self.optimizer[1]([states, discounted_rewards])
        self.optimizer[0]([states, actions, advantages])


def save_history(history, name):
    """save reward history.
    """
    name = os.path.join('history', name)

    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')


def play(model):
    """play game with model.
    """
    print('play...')

    env = gym.make('CartPole-v0')
    observation = env.reset()

    reward_sum = 0
    random_episodes = 0

    while random_episodes < 10:
        env.render()

        prob = model.actor.predict(observation.reshape(-1, 4))[0][0]
        action = 1 if prob > 0.5 else 0

        observation, reward, done, _ = env.step(action)

        reward_sum += reward

        if done:
            print("Reward for this episode was: {}".format(reward_sum))
            random_episodes += 1
            reward_sum = 0
            observation = env.reset()

    env.close()


if __name__ == '__main__':
    model = A3C()
#
#    model.train(2000, 4, 10)
#    save_history(history, 'a3c_sparse.csv')

    model.load()
    play(model)
