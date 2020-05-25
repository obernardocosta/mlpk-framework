from algorithms.model import Model

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
import numpy as np
from math import sqrt


class LightGBM(Model):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1, silent=True, importance_type='split', kwargs={}, metrics = [], path = 'algorithms/.output', name = 'lightgbm'):
        
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        self.kwargs = kwargs
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = LGBMClassifier(boosting_type=self.boosting_type,
                                    num_leaves=self.num_leaves,
                                    max_depth=self.max_depth,
                                    learning_rate=self.learning_rate,
                                    n_estimators=self.n_estimators,
                                    subsample_for_bin=self.subsample_for_bin,
                                    objective=self.objective,
                                    class_weight=self.class_weight,
                                    min_split_gain=self.min_split_gain,
                                    min_child_weight=self.min_child_weight,
                                    min_child_samples=self.min_child_samples,
                                    subsample=self.subsample,
                                    subsample_freq=self.subsample_freq,
                                    colsample_bytree=self.colsample_bytree,
                                    reg_alpha=self.reg_alpha,
                                    reg_lambda=self.reg_lambda,
                                    random_state=self.random_state,
                                    n_jobs=self.n_jobs,
                                    silent=self.silent,
                                    importance_type=self.importance_type,
                                    **self.kwargs)


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
