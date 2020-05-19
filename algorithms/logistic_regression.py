from algorithms.model import Model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
from math import sqrt


class LogisticRegressionBCls(Model):
    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, metrics=[], path='algorithms/.output', name='logistic_regression_bcls'):
        
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.metrics = metrics
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = LogisticRegression(penalty=self.penalty,
                                        dual=self.dual,
                                        tol=self.tol,
                                        C=self.C,
                                        fit_intercept=self.fit_intercept,
                                        intercept_scaling=self.intercept_scaling,
                                        class_weight=self.class_weight,
                                        random_state=self.random_state,
                                        solver=self.solver,
                                        max_iter=self.max_iter,
                                        multi_class=self.multi_class,
                                        verbose=self.verbose,
                                        warm_start=self.warm_start,
                                        n_jobs=self.n_jobs,
                                        l1_ratio=self.l1_ratio)


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
