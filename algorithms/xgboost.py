from algorithms.model import Model

from xgboost import XGBModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
import numpy as np
from math import sqrt


class XGBoost(Model):
    def __init__(self, objective="binary:logistic", max_depth=None, learning_rate=None, n_estimators=100, verbosity=None, booster=None, tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None, subsample=None, colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None, monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None, metrics=[], path='algorithms/.output', name='xgboost'):
        
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = XGBClassifier(objective=self.objective,
                              max_depth=self.max_depth,
                              learning_rate=self.learning_rate,
                              n_estimators=self.n_estimators,
                              verbosity=self.verbosity,
                              booster=self.booster,
                              tree_method=self.tree_method,
                              n_jobs=self.n_jobs,
                              gamma=self.gamma,
                              min_child_weight=self.min_child_weight,
                              max_delta_step=self.max_delta_step,
                              subsample=self.subsample,
                              colsample_bytree=self.colsample_bytree,
                              colsample_bylevel=self.colsample_bylevel,
                              colsample_bynode=self.colsample_bynode,
                              reg_alpha=self.reg_alpha,
                              reg_lambda=self.reg_lambda,
                              scale_pos_weight=self.scale_pos_weight,
                              base_score=self.base_score,
                              random_state=self.random_state,
                              missing=self.missing,
                              num_parallel_tree=self.num_parallel_tree,
                              monotone_constraints=self.monotone_constraints,
                              interaction_constraints=self.interaction_constraints,
                              importance_type=self.importance_type,
                              gpu_id=self.gpu_id,
                              validate_parameters=self.validate_parameters)


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
