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
        self.batch_size = 64
        
        # hyper - params
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.gamma = 0.95   # discout rate

        self.model = self.get_init_model()
        self.memory = deque(maxlen=100000)     # replay memory

        # Eval
        self.val_set = None
        self.total_reward_per_epoch = []
        self.val_score_per_epoch = []

        # Environment
        self.env = Paddle()


    def get_init_model(self) -> Model:
        """Get model with initialized weights for the layers
        """
        # Feature inputs:
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
        optimizer = optimizers.Adam(learning_rate=0.01)

        model.compile(
            loss=losses.MeanSquaredError(), 
            optimizer=optimizer, 
            metrics=[metrics.MeanSquaredError()])
        
        model.summary()
        return model


    def is_random_action_policy(self):
        dice = random.uniform(0, 1)
        is_random = dice < self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return is_random


    def collect_validation_set(self):
        # run random action policy before training and collect samples for validation set
        print("Playing random policy game to collect validation set")
        step_counter = 0
        val_set = []

        while step_counter < 1000:
            state = self.env.get_state()
            action = random.randint(0, 2)   # random action
            reward, new_state, done = self.env.step(action)  # execute action
            val_set.append({
                'state': state,
                'action': action,
                'reward': reward,
                'new_state': new_state,
                'done': done
            })
            step_counter += 1

        self.val_set = np.array(val_set)


    def train(self, epochs=1000):
        games_counter = 0
        steps_counter = 0
        total_game_reward = 0
        val_scores = []

        while games_counter < epochs:
            state = self.env.get_state()

            # random action policy
            if self.is_random_action_policy():
                action = random.randint(0,2)
            # greedy action policy
            else:
                action = self.select_action(state)

            # execute action in env
            reward, new_state, done = self.env.step(action)
            total_game_reward += reward

            # replay memory
            self.memory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'new_state': new_state,
                    'done': done
                })

            if steps_counter % 20 == 0:
                val_score = self.update_weights()
                val_scores.append(val_score)

            steps_counter += 1
            if done:
                games_counter += 1
                self.total_reward_per_epoch.append(total_game_reward)
                self.val_score_per_epoch.append(np.avg(val_scores))
                # reset for next epoch
                total_game_reward = 0
                val_scores = []


    def get_x_y_from_replays(self, replays):
        """
        Generate X, Y variables for model training from replays
        """
        X = np.array([r['state'] for r in replays]).reshape(-1, 5)

        actions = np.array([r['action'] for r in replays])
        rewards = np.array([r['reward'] for r in replays])
        dones = np.array([r['done'] for r in replays])

        X_next = np.array([r['new_state'] for r in replays]).reshape(-1,5)
        next_rewards = np.apply_along_axis(np.max, 1, self.model.predict_on_batch(X_next))
        Y = rewards + self.gamma * next_rewards * (1-dones)

        #### *** Setting target for training ***
        Y_full = self.model.predict_on_batch(X)     # predicted rewards / action for current weights
        ix = np.arange(self.batch_size)
        Y_full[ix , actions] = Y            # replace target reward for the selected action from sample for training

        return X, Y_full


    def update_weights(self):
        if len(self.memory) < self.batch_size:
            return

        # sample from memory
        replays = random.sample(self.memory, k=self.batch_size)
        X, Y_full = self.get_x_y_from_replays(replays)

        # update weights SGD
        self.model.fit(
            x=X,
            y=Y_full,
            epochs=1,
            verbose=0
        )

        # Evaluation val set
        X_val = np.array([r['state'] for r in self.val_set]).reshape(-1, 5)
        Y_val_pred = np.apply_along_axis(np.max, 1, self.model.predict_on_batch(X_val))
        total_val_reward = np.sum(Y_val_pred)
        return total_val_reward


    def predict_rewards(self, state):
        x = np.array(state).reshape(-1, len(state))
        rewards = self.model.predict([x])[0]
        return rewards

    def select_action(self, state):
        rewards = self.predict_rewards(state)
        action = np.argmax(rewards)
        return action


    # def get_model():
    #     pass
    #
    #
    # def save_model():
    #     pass
    #
    #
    # def load_model():
    #     pass



if __name__ == "__main__":
    agent = Agent()
    agent.train()
