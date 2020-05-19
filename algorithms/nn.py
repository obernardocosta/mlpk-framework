from algorithms.model import Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


class NN(Model):
    def __init__(self, n_layers=1, input_dim=1, n_neurons=[], list_act_func=[], loss='', optimizer='', metric=[], path='algorithms/.output', name='nn'):
        
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.list_act_func = list_act_func
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = keras.models.Sequential(name=self.name)

        for layer, act_f, n in zip(range(1, self.n_layers+1), self.list_act_func, self.n_neurons):
            if layer == 1:
                self.model.add(layers.Dense(
                    n, input_dim=self.input_dim, activation=act_f, name="layer{}".format(layer)))
            else:
                self.model.add(layers.Dense(
                    n, activation=act_f, name="layer{}".format(layer)))
        
        self.create_history_folder(self.path, self.name)

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metric])


    def train(self, X_train, y_train, epochs):

        X_train, X_validation = train_test_split(X_train, test_size=0.1)
        y_train, y_validation = train_test_split(y_train, test_size=0.1)


        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_validation, y_validation))
        
        self._plot_history(self.history, self.path, self.name)
        
        return self.history


    def evaluate(self, X_test, y_test):
        self.scores = self.model.evaluate(X_test, y_test)
        return self.scores
    

    def predict(self, X_new):
        self.scores = self.model.predict(X_new)
        return self.scores
        

    def save(self, model_name):
        self.model.save(model_name)
    
    
    def load(self, model_name):
        self.model = tf.keras.models.load_model(model_name)
