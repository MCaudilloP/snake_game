import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Modelo del agente
class SnakeAgent:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.input_dim,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.output_dim, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def train(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def get_action(self, state):
        # Obtener la acción con la mayor probabilidad del modelo
        state = np.array(state)
        state = state.reshape(1, -1)
        action_probs = self.model.predict(state)[0]
        action = np.argmax(action_probs)
        return action

    def q_learning_train(self, states, actions, rewards):
        # Entrenar el modelo con los estados, acciones y recompensas proporcionados
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Codificar las acciones como one-hot vectors
        actions_onehot = np.zeros((len(actions), 4))
        actions_onehot[np.arange(len(actions)), actions] = 1

        # Actualizar el modelo utilizando el algoritmo Q-learning
        self.model.fit(states, actions_onehot, sample_weight=rewards, epochs=1, verbose=0)
