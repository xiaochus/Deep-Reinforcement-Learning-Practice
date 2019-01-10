# -*- coding: utf-8 -*-
import os
import random
import numpy as np

from DQN import DQN


class DDQN(DQN):
    """Nature Deep Q-Learning.
    """
    def __init__(self):
        super(DDQN, self).__init__()

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def load(self):
        if os.path.exists('model/ddqn.h5'):
            self.model.load_weights('model/ddqn.h5')

    def update_target_model(self):
        """update target_model
        """
        self.target_model.set_weights(self.model.get_weights())

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
        q = self.target_model.predict(next_states)
        next_action = np.argmax(self.model.predict(next_states), axis=1)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * q[i][next_action[i]]
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

                    # update target_model every 20 episode
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))

        self.model.save_weights('model/ddqn.h5')

        return history


if __name__ == '__main__':
    model = DDQN()

    history = model.train(600, 32)
    model.save_history(history, 'ddqn.csv')

    model.load()
    model.play('dqn')
