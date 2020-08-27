import time
import shutil

import numpy as np
import pandas as pd

from datetime import timedelta
from datetime import datetime as dt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from algorithms.nn import NN
from algorithms.svm import SVM
from algorithms.xgboost import XGBoost
from algorithms.lightgbm import LightGBM
from algorithms.decision_tree import DecisionTree
from algorithms.random_forest import RandomForest
from algorithms.naive_bayes_bcls import NaiveBayesBCls
from algorithms.logistic_regression import LogisticRegressionBCls


class Runner:
    """Base Class for Machine Learning methods."""

    def __init__(self, algo_conf, n_split, n_components, test_size, start_date, data_path, columns, epochs=0, shuffle=True):

        self._ALGORITHMS_OUTPUT_PATH = './algorithms/.output'
        
        self.algo_conf = algo_conf
        self.epochs = epochs
        self.n_split = n_split
        self.n_components = n_components
        self.test_size = test_size
        self.start_date = start_date
        self.data_path = data_path
        self.columns = columns
        self.metric = self.algo_conf['metric']
        self.algo_conf.pop('metric', None)
        self.shuffle = shuffle

        
        self.feature_columns = [x for x in self.columns if x != 'y']
        self.pre_models = []

    
    def run(self):

        start = time.time()
        print('Runner init', start)

        shutil.rmtree(self._ALGORITHMS_OUTPUT_PATH, ignore_errors=True)

        self.df = Runner.load_data(path=self.data_path, names=self.columns)
        self.df = Runner.add_data(self.df, start_date=self.start_date)

        self.skfold = StratifiedKFold(
            n_splits=self.n_split, shuffle=True, random_state=None)

        self.Train, self.Test = train_test_split(
            self.df, test_size=self.test_size, shuffle=self.shuffle)
        
        self.choose_best_kf_model()
        self.run_for_the_best_model()

        end = time.time()
        self.timer = end - start
        print('Runner finish. Timer:', self.timer)


    def run_for_the_best_model(self):

        seq = [x['accuracy'] for x in self.pre_models]
        max_ = max(seq)

        print('---------------------------')
        print('best model {}: {}'.format(self.metric, max_))
        print('---------------------------')

        _model = [
            model for model in self.pre_models if model[self.metric] == max_][0]
        model = _model['model']

        X_train = self.Train[self.feature_columns]
        y_train = self.Train[['y']]
        X_test = self.Test[self.feature_columns]
        y_test = self.Test[['y']]

        means = dict(np.mean(X_train))
        stds = dict(np.std(X_train))

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        Train = Runner.pca_transformation(X_train, n_components=self.n_components)
        X_test = Runner.pca_transformation(X_test, n_components=self.n_components)

        history = model.train(X_train, y_train, epochs=self.epochs)
        scores = model.evaluate(X_test, y_test)

        print("rms %.4f%% %s: %.4f%%" %
            (scores[0] * 100, self.metric, scores[1] * 100))

        self.result_metric = {
            'rms': scores[0] * 100,
            self.metric: scores[1] * 100,
            'means': means,
            'stds': stds,
            'model': model
        }

        print('---------------------------')
        print('pre_models', self.pre_models)
        print('---------------------------')
        print('---------------------------')
        print('result for the best model',  self.result_metric)
        print('---------------------------')


    def choose_best_kf_model(self):
        for train_index, test_index in self.skfold.split(self.Train.loc[:, self.Train.columns != 'y'], self.Train['y']):

            X_train = self.df.loc[train_index]
            y_train = self.df['y'].loc[train_index]
            X_evaluate = self.df.loc[test_index]
            y_evaluate = self.df['y'].loc[test_index]

            means = dict(np.mean(X_train[self.feature_columns]))
            stds = dict(np.std(X_train[self.feature_columns]))

            scaler = StandardScaler().fit(X_train[self.feature_columns])
            X_train[self.feature_columns] = scaler.transform(
                X_train[self.feature_columns])

            X_train = Runner.pca_transformation(
                X_train, n_components=self.n_components)
            X_evaluate = Runner.pca_transformation(
                X_evaluate, n_components=self.n_components)

            model = self.get_model()

            history = model.train(X_train, y_train, epochs=self.epochs)
            scores = model.evaluate(X_evaluate, y_evaluate)
            print("rms %.4f%% %s: %.4f%%" %
                (scores[0] * 100, self.metric, scores[1] * 100))

            self.pre_models.append(
                {
                    'rms': scores[0] * 100,
                    self.metric: scores[1] * 100,
                    'means': means,
                    'stds': stds,
                    'model': model
                }
            )


    def get_model(self):
        model = None
        name = self.algo_conf['name']
        
        if name == 'NN':
            self.algo_conf['metric'] = self.metric
            model = NN(**self.algo_conf)
        elif name == 'naive_bayes_bcls':
            model = NaiveBayesBCls(**self.algo_conf)
        elif name == 'logistic_regression_bcls':
            model = LogisticRegressionBCls(**self.algo_conf)
        elif name == 'svm':
            model = SVM(**self.algo_conf)
        elif name == 'decision_tree':
            model = DecisionTree(**self.algo_conf)
        elif name == 'random_forest':
            model = RandomForest(**self.algo_conf)
        elif name == 'xgboost':
            model = XGBoost(**self.algo_conf)
        elif name == 'lightgbm':
            model = LightGBM(**self.algo_conf)
        elif name == 'LSTM':
            model = LSTM(**self.algo_conf)



        return model


    @staticmethod
    def load_data(path='', names=None):
        print(path, names)
        print('****')
        df = pd.read_csv(path, index_col=None, names=names,
                        sep=', ', engine='python')
        return df


    @staticmethod
    def add_data(df, start_date=''):

        date = dt.strptime(start_date, "%Y/%m/%d")
        df['index'] = np.arange(len(df))
        df['date'] = df['index'].apply(lambda x: date + timedelta(days=int(x)))
        df['day'], df['month'], df['year'] = df['date'].dt.day, df['date'].dt.month, df['date'].dt.year

        del df['index']
        del df['date']

        return df


    @staticmethod    
    def get_now():
        return dt.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


    @staticmethod
    def pca_transformation(X, n_components):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca