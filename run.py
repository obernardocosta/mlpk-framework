from Runner import Runner as R


def main():

    '''mandatory parameters'''
    n_split = 10
    n_components = 2
    test_size = 0.2
    start_date = '2013/08/19'
    data_path = './data/final-mix/data-1.csv'
    columns = ['btc', 'gt', 'y']
    epochs = 1

    '''For Neural Nets'''
    '''
    algo_conf = {
        'algorithms_name': 'NN',
        'n_layers': 3,
        'input_dim': 2,
        'n_neurons': [5, 8, 1],
        'list_act_func': ['tanh', 'tanh', 'sigmoid'],
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metric': 'accuracy'
    }
    '''

    '''For Naive Bayes Binary CLs'''
    algo_conf = {
        'algorithms_name': 'naive_bayes_bcls',
        'priors': None,
        'var_smoothing': 0.1,
        'metric': 'accuracy'
    }
    

    r = R(algo_conf=algo_conf, epochs=epochs, n_split=n_split, n_components=n_components,
          test_size=test_size, start_date=start_date, data_path=data_path, columns=columns)

    print('Runner init', r.get_now())
    r.run()
    print('Runner end', r.get_now())


if __name__ == '__main__':
    main()   
