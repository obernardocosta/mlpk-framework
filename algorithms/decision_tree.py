from algorithms.model import Model

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
from math import sqrt


class DecisionTree(Model):
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0, metrics=[], path='algorithms/.output', name='decision_tree'):
        
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
        self.ccp_alpha = ccp_alpha
        self.metrics = metrics
        self.path = path
        self.name = name

        self.create_model()

        super().__init__(self.path, self.name)
    

    def create_model(self):
        self.model = tree.DecisionTreeClassifier(criterion=self.criterion,
                                                 splitter=self.splitter,
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                 max_features=self.max_features,
                                                 random_state=self.random_state,
                                                 max_leaf_nodes=self.max_leaf_nodes,
                                                 min_impurity_decrease=self.min_impurity_decrease,
                                                 min_impurity_split=self.min_impurity_split,
                                                 class_weight=self.class_weight,
                                                 presort=self.presort,
                                                 ccp_alpha=self.ccp_alpha)


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
