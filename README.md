# MLPK Framework

**M**achine **L**earning **P**CA **K**-Fold Framework

## Tutorial

1. Set the mandatory parameters and the choosen algorithm parameters in the run.py
2. ```python3 run.py```

### Mandatory Parameters

- n_split: K of K-fold
- n_components: PCA number of components
- test_size: percentage of data used in the test part (0<test_size<1)
- start_date: date in in string format to add in dataset
- data_path: dataset path
- columns: columns of dataset (must have a **y** column)
- epochs: number of epochs

### Algorithms Parameters

#### Neural Networks

    algo_conf = {
        'name': 'NN',
        'n_layers': 3,
        'input_dim': 2,
        'n_neurons': [5, 8, 1],
        'list_act_func': ['tanh', 'tanh', 'sigmoid'],
        'loss': 'mean_squared_error',
        'optimizer': 'sgd',
        'metric': 'accuracy'
    }

#### Naive Bayes Binary Cls

    algo_conf = {
        'name': 'naive_bayes_bcls',
        'priors': None,
        'var_smoothing': 0.1,
        'metric': 'accuracy'
    }

#### Logistic Regression Binary Cls

    algo_conf = {
        'name': 'logistic_regression_bcls',
        'solver': 'liblinear',
        'metric': 'accuracy'
    }

#### SVM  Binary Cls

    algo_conf = {
        'name': 'svm',
        'metric': 'accuracy'
    }

#### Decision Tree Binary Cls

    algo_conf = {
        'name': 'decision_tree',
        'metric': 'accuracy'
    }

#### Random Forest Binary Cls

    algo_conf = {
        'name': 'random_forest',
        'metric': 'accuracy'
    }

#### XGBoot Binary Cls

    algo_conf = {
        'objective': 'binary:logistic',
        'name': 'xgboost',
        'metric': 'accuracy'
    }

#### LightGBM Binary Cls

    algo_conf = {
        'name': 'lightgbm',
        'metric': 'accuracy'
    }    

## Sample data in ./data

It's just a sample dataset, you can adapt your own dataset.

- Bitcoin: 2013-08-19 to 2016-07-19
  - Data market price from [charts.bitcoin](https://charts.bitcoin.com/bch/)

## References

- [TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
- [Neural Network on Keras](https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)
- [PCA on Keras](https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network)
- [Adam Optimization](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- [Google Trends Normalization code](https://github.com/maliky)
