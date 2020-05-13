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


def load_data(path='', names=None):
    df = pd.read_csv(path, index_col=None, names=names, sep=', ', engine='python')
    return df

def add_data(df, start_date=''):
    
    date = dt.strptime(start_date, "%Y/%m/%d")
    df['index'] = np.arange(len(df))
    df['date'] = df['index'].apply(lambda x: date + timedelta(days=int(x)))
    df['day'], df['month'], df['year'] = df['date'].dt.day, df['date'].dt.month, df['date'].dt.year

    del df['index']
    del df['date']
    
    return df


def get_now():
    return dt.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def pca_transformation(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca


def main():

    print('Runner init', get_now())
    #shutil.rmtree('./algorithms/.output')
    df = load_data(path='./data/final-mix/data-1.csv', names=['btc', 'gt', 'y'])
    #df = add_data(df, start_date="2013/08/19")
    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    Train, Test = train_test_split(df, test_size=0.2)

    for train_index, test_index in skfold.split(Train.loc[:, Train.columns != 'y'], Train['y']):
        X_train = df.loc[train_index]
        y_train = df['y'].loc[train_index]

        X_evaluate = df.loc[test_index]
        y_evaluate = df['y'].loc[test_index]

        means = dict(np.mean(X_train[['btc', 'gt']]))
        stds = dict(np.std(X_train[['btc', 'gt']]))
        print(means, stds)

        scaler = StandardScaler().fit(X_train[['btc', 'gt']])
        X_train[['btc', 'gt']] = scaler.transform(X_train[['btc', 'gt']])

        n = 2
        X_train = pca_transformation(X_train, n_components=n)
        X_evaluate = pca_transformation(X_evaluate, n_components=n)

        # Alg part
        model = NN(n_layers=3,
                      input_dim=2,
                      n_neurons=[5, 8, 1],
                      list_act_func=['tanh', 'tanh', 'sigmoid'],
                      name='nn',
                      loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])


        history = model.train(X_train, y_train, epochs=100)
        scores = model.evaluate(X_evaluate, y_evaluate)
        print("rms %.4f%% %s: %.4f%%" %
              (scores[0] * 100, model.model.metrics_names[1], scores[1] * 100))
       # alg end


if __name__ == '__main__':
    main()   
