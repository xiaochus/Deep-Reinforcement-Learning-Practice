# -*- coding: utf-8 -*-
import os
import numpy as np

from keras.layers import Input, Dense, Add, Subtract, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from NatureDQN import NDQN


class DuelingDQN(NDQN):
    """Dueling DQN.
    """
    def __init__(self):
        super(DuelingDQN, self).__init__()

    def load(self):
        if os.path.exists('model/dueling.h5'):
            self.model.load_weights('model/dueling.h5')

    def build_model(self):
        """basic model.
        """
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)

        value = Dense(2, activation='linear')(x)
        a = Dense(2, activation='linear')(x)
        meam = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, meam])

        q = Add()([value, advantage])

        model = Model(inputs=inputs, outputs=q)

        model.compile(loss='mse', optimizer=Adam(1e-3))

        return model

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

        self.model.save_weights('model/dueling.h5')

        return history


if __name__ == '__main__':
    model = DuelingDQN()

    history = model.train(600, 32)
    model.save_history(history, 'dueling.csv')

    model.load()
    model.play('dqn')
