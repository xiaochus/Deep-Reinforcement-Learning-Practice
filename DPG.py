# -*- coding: utf-8 -*-
import os

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from DRL import DRL


class DPG(DRL):
    """Deterministic Policy Gradient Algorithms
    """
    def __init__(self):
        super(DPG, self).__init__()

        self.model = self._build_model()

        if os.path.exists('model/dpg.h5'):
            self.model.load_weights('model/dpg.h5')

        self.gamma = 0.95

    def _build_model(self):
        """basic model.
        """
        inputs = Input(shape=(4,), name='ob_input')
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def loss(self, y_true, y_pred):
        """loss function.
        Arguments:
            y_true: (action, reward)
            y_pred: action_prob

        Returns:
            loss: reward loss
        """
        action_pred = y_pred
        action_true, discount_episode_reward = y_true[:, 0], y_true[:, 1]

        action_true = K.reshape(action_true, (-1, 1))
        loss = K.binary_crossentropy(action_true, action_pred)
        loss = loss * K.flatten(discount_episode_reward)

        return loss

    def discount_reward(self, rewards):
        """Discount reward
        Arguments:
            rewards: rewards in a episode.
        """
        # compute the discounted reward backwards through time.
        discount_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discount_rewards[i] = cumulative

        # size the rewards to be unit normal (helps control the gradient estimator variance).
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards //= np.std(discount_rewards)

        return list(discount_rewards)

    def train(self, episode, batch):
        """training model.
        Arguments:
            episode: ganme episode
            batch： batch size of episode

        Returns:
            history: training history
        """
        self.model.compile(loss=self.loss, optimizer=Adam(lr=0.01))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        episode_reward = 0
        states = []
        actions = []
        rewards = []
        discount_rewards = []

        for i in range(episode):
            observation = self.env.reset()
            erewards = []

            while True:
                x = observation.reshape(-1, 4)
                prob = self.model.predict(x)[0][0]

                # choice action with prob.
                action = np.random.choice(np.array(range(2)), size=1, p=[1 - prob, prob])[0]
                observation, reward, done, _ = self.env.step(action)

                states.append(x[0])
                actions.append(action)
                erewards.append(reward)
                rewards.append(reward)

                if done:
                    # calculate discount rewards every episode.
                    discount_rewards.extend(self.discount_reward(erewards))
                    break

            if i != 0 and i % batch == 0: 
                episode_reward = sum(rewards) / batch

                X = np.array(states)
                y = np.array(list(zip(actions, discount_rewards)))

                loss = self.model.train_on_batch(X, y)

                history['episode'].append(i)
                history['Episode_reward'].append(episode_reward)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f}'.format(i, episode_reward, loss))

                episode_reward = 0
                states = []
                actions = []
                rewards = []
                discount_rewards = []

        self.model.save_weights('model/dpg.h5')

        return history


if __name__ == '__main__':
    model = DPG()

    history = model.train(5000, 5)
    model.save_history(history, 'dpg.csv')

    model.play()
