from algorithms.model import Model

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
from math import sqrt


class NaiveBayesBCls(Model):
    def __init__(self, priors=None, var_smoothing=1e-9, metrics=[], path='algorithms/.output', name='naive_bayes_bcls'):
        
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.metrics = metrics
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = GaussianNB(priors=self.priors, var_smoothing=self.var_smoothing)


    def train(self, X_train, y_train, epochs):

        self.model.fit(X_train, y_train)
        return None


    def evaluate(self, X_test, y_test):
        yhat = self.predict(X_test)
        self.scores = [
            sqrt(mean_squared_error(y_test, yhat)),
            accuracy_score(y_test, yhat)
        ]
        return self.scores
    

    def predict(self, X_new):
        self.yhat = self.model.predict(X_new)
        return self.yhat
        

    def save(self, model_name):
        with open(model_name, 'wb') as f:
            pickle.dump(self.model, f)
    
    
    def load(self, model_name):
        with open(model_name, 'rb') as f:
            self.model = pickle.load(f)
