# -*- coding: utf-8 -*-
import os
import random
import numpy as np

from collections import deque

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from DRL import DRL


class DQN(DRL):
    """Deep Q-Learning.
    """
    def __init__(self):
        super(DQN, self).__init__()

        self.model = self.build_model()

        if os.path.exists('model/dqn.h5'):
            self.model.load_weights('model/dqn.h5')

        # experience replay.
        self.memory_buffer = deque(maxlen=2000)
        # discount rate for q value.
        self.gamma = 0.95
        # epsilon of ε-greedy.
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01


    def build_model(self):
        """basic model.
        """
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='mse', optimizer=Adam(1e-3))

        return model

    def egreedy_action(self, state):
        """ε-greedy
        Arguments:
            state: observation

        Returns:
            action: action
        """
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.
        Arguments:
            state: observation
            action: action
            reward: reward
            next_state: next_observation
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """process batch data
        Arguments:
            batch: batch size

        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        # Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y


    def train(self, episode, batch):
        """training 
        Arguments:
            episode: game episode
            batch： batch size

        Returns:
            history: training history
        """
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # chocie action from ε-greedy.
                x = observation.reshape(-1, 4)
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action)
                # add data to experience replay.
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # reduce epsilon pure batch.
                    self.update_epsilon()

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))

        self.model.save_weights('model/dqn.h5')

        return history


if __name__ == '__main__':
    model = DQN()

    history = model.train(600, 32)
    model.save_history(history, 'dqn.csv')

    model.play()
