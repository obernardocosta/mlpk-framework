from Runner import Runner as R


def main():

    '''Mandatory Parameters'''
    n_split = 10
    n_components = 6
    test_size = 0.2
    start_date = '2013/08/19'
    data_path = './data/final-mix/data-3.csv'
    columns = ['btc1','btc2', 'btc3', 'gt1', 'gt2', 'gt3', 'y']
    epochs = 100
    
    '''Algorithms Parameters'''
    algo_conf = {
        'name': 'lightgbm',
        'metric': 'accuracy'
    } 
    
    
    r = R(algo_conf=algo_conf,
          epochs=epochs,
          n_split=n_split,
          n_components=n_components,
          test_size=test_size,
          start_date=start_date,
          data_path=data_path,
          columns=columns)

    r.run()


if __name__ == '__main__':
    main()   
