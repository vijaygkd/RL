import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics, losses

from environments.paddle.paddle import Paddle


class Agent:

    def __init__(self):
        self.epochs = 1000

        self.model = self.get_init_model()
        self.D = []     # replay memory

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


    def train(self):
        games_counter = 0
        steps_counter = 0

        while games_counter < self.epochs:
            state = self.env.get_state()

            # greedy or random policy
            action = self.select_action(state)

            # execute action in env
            reward, new_state, done = self.env.step(action)

            # replay memory
            self.D.append((state, action, reward, new_state))

            # sample from D

            # set y for each lable

            # update weights SGD

            steps_counter += 1
            if done:
                games_counter += 1


    def update_weights():
        pass


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
