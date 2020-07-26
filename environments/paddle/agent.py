import random
from collections import deque

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics, losses

from environments.paddle.paddle import Paddle


class Agent:

    def __init__(self):
        self.epochs = 1000
        self.batch_size = 64
        
        # hyper - params
        self.epsilon = 1
        self.epsilon_decay = 0.995

        self.gamma = 0.95   # discout rate

        self.model = self.get_init_model()
        self.memory = deque(maxlen=100000)     # replay memory

        self.env = Paddle()


    def get_init_model(self) -> Model:
        """Get model with initialized weights for the layers

        Returns:
            Model -- Keras model with initialized layer weights
        """
        # Feature inputs:
        # state
        input_layer = Input(shape=(5,), name='input_features')

        # Hidden layers
        hidden_1 = Dense(64, activation='relu', name='hidden_1')(input_layer)
        hidden_2 = Dense(32, activation='relu', name='hidden_2')(hidden_1)
        # Output layer
        # output is reward for each action 0-stay, 1-left, 2-right
        output_layer = Dense(3, activation='linear', name='output')(hidden_2)

        # Model
        model = Model(inputs=[input_layer],
                      outputs=output_layer)

        # Optimizer
        optimizer = optimizers.Adam(learning_rate=0.001)

        model.compile(
            loss=losses.MeanSquaredError(), 
            optimizer=optimizer, 
            metrics=[metrics.MeanSquaredError()])
        
        model.summary()
        return model


    def is_random_action_policy(self):
        dice = random.uniform(0, 1)
        is_random = dice < self.epsilon
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay
        return is_random


    def train(self):
        games_counter = 0
        steps_counter = 0

        while games_counter < self.epochs:
            state = self.env.get_state()

            # random action policy
            if self.is_random_action_policy():
                action = random.randint(0,2)
            # greedy action policy
            else:
                action = self.select_action(state)

            # execute action in env
            reward, new_state, done = self.env.step(action)

            # replay memory
            self.memory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'new_state': new_state,
                    'done': done
                })

            if steps_counter % 20 == 0:
                self.update_weights()

            steps_counter += 1
            if done:
                games_counter += 1

    def update_weights(self):
        if len(self.memory) < self.batch_size:
            return

        # sample from D
        replays = random.sample(self.memory, k=self.batch_size)
        X = np.array([r['state'] for r in replays])
        X = X.reshape(-1, 5)

        # def set_y(replay):
        #     action = replay['action']
        #     if replay['done']:
        #         r = replay['reward']
        #     else:
        #         next_reward = self.predict_rewards(replay['new_state'])
        #         max_next_reward = np.max(next_reward)
        #         r = replay['reward'] + self.gamma * max_next_reward
        #     y[action] = r
        #     return y
        #
        # Y = list(map(set_y, replays))
        # Y = np.array(Y).reshape(-1, 3)  # one-hot y value


        actions = np.array([r['action'] for r in replays])
        rewards = np.array([r['reward'] for r in replays])
        dones = np.array([r['done'] for r in replays])

        X_next = np.array([r['new_state'] for r in replays]).reshape(-1,5)
        next_rewards = self.model.predict_on_batch(X_next).apply_along_axis(np.max, 1)
        Y = rewards + self.gamma * next_rewards * (1-dones)

        Y_full = self.model.predict_on_batch(X)
        Y_full[actions] = Y

        # update weights SGD
        self.model.fit(
            x=X,
            y=Y_full,
            epochs=1
        )


    def predict_rewards(self, state):
        x = np.array(state).reshape(-1, len(state))
        rewards = self.model.predict([x])[0]
        return rewards

    def select_action(self, state):
        rewards = self.predict_rewards(state)
        action = np.argmax(rewards)
        return action


    def get_model():
        pass


    def save_model():
        pass


    def load_model():
        pass



if __name__ == "__main__":
    agent = Agent()
    agent.train()
