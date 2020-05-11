import os
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


class NN:
    def __init__(self, n_layers=1, input_dim=1, n_neurons=[], list_act_func=[], model_name='my_model', loss='', optimizer='', metrics=[]):

        self.model = keras.models.Sequential(name=model_name)

        for l, act_f, n in zip(range(1, n_layers+1), list_act_func, n_neurons):
            print(l, act_f, n)
            if l == 1:
                self.model.add(layers.Dense(
                    n, input_dim=input_dim, activation=act_f, name="layer{}".format(l)))
            else:
                self.model.add(layers.Dense(n, activation=act_f, name="layer{}".format(l)))
        
        print(self.model.summary())

        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)
    

    def train(self, X_train, y_train, epochs):

        X_train, X_validation = train_test_split(X_train, test_size=0.1)
        y_train, y_validation = train_test_split(y_train, test_size=0.1)


        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_validation, y_validation))
        
        #self._plot_history(self.history)


    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test)
    

    def predict(self, X_new):
        return self.model.predict(X_new)
        

    def save(self, model_name):
        self.model.save(model_name)
    
    
    def load(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    @staticmethod
    def _plot_history(history):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
