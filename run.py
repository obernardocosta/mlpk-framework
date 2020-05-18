from Runner import Runner as R


def main():

    n_split = 10
    n_components = 2
    test_size = 0.2
    start_date = '2013/08/19'
    data_path = './data/final-mix/data-1.csv'
    columns = ['btc', 'gt', 'y']

    '''For Neural Nets'''
    '''
    epochs = 1
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
    ''''


    r = R(algo_conf, epochs, n_split, n_components, test_size, start_date, data_path, columns)
    
    print('Runner init', r.get_now())
    r.run()
    print('Runner end', r.get_now())


if __name__ == '__main__':
    main()   
