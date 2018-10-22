import numpy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys
import csv

seed = None
numpy.random.seed()

#normalizing data for preprocessing
def mean_std(x, mean, std):
    return float((x-mean)/std)

def pca_train_data_generator(X, pca):
    X_pca_train = pca.fit_transform(X)
    pca_std = numpy.std(X_pca_train)
    return X_pca_train, pca_std

def pca_test_generator(X, pca):
    X_pca_test = pca.transform(X)
    return X_pca_test

def pca_predict_generator(X, pca):
    X_pca_test = pca.transform(X)
    return X_pca_test

def mean_std_transform(x_train, X_test):
    train_t = numpy.transpose(x_train)
    test_t = numpy.transpose(X_test)

    train = []
    train_c = []
    test = []
    test_c = []
    means = []
    stds = []

    for t in train_t:
        mean, std = numpy.mean(t), numpy.std(t)
        means.append(mean)
        stds.append(std)
        for x in t:
            train_c.append(mean_std(x, mean, std))
        train.append(train_c)
        train_c = []

    for i in range(len(test_t)):

        for j in range(len(test_t[i])):
            test_c.append(mean_std(test_t[i][j], means[i], stds[i]))
        test.append(test_c)
        test_c = []

    return numpy.transpose(train), numpy.transpose(test)


def fina_mix_pca(data_type, i):

    x = 1
    if data_type == "final-mix":
        x = i

    return x

def save_csv(_csv):
    with open("data.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in _csv:
            writer.writerow([val])

def main():

    data_type = sys.argv[1]
    range_data_min = int(sys.argv[2])
    range_data_max = int(sys.argv[3])


    for i in range(range_data_min,range_data_max):

        x = fina_mix_pca(data_type, i)

        for j in range(i*x):

            k = j+1
            l = 1

            if data_type == "final-mix":
                l = 2
                k = k * 2


            if k > i*2:
                break

            print("pca with n=", k)
            number_of_folds = k

            #load PCA
            pca = PCA(n_components=number_of_folds)

            ## load and prepare the dataset
            path ="./../data/"+data_type+"/data-"+str(i)+".csv"
            print("data",str(i))



            dataset = numpy.loadtxt(path, delimiter=",")
            X = dataset[:,0:i*l]
            y = dataset[:,i*l]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            X_train, X_test = mean_std_transform(X_train, X_test)

            #save_csv(X_train)


            X_pca_train, pca_std = pca_train_data_generator(X_train, pca)
            print(len(X_pca_train[0]))



            # 0. define 10-fold cross validation test harness
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            cvscores = []
            cvmse = []
            for neuron_number in range(1,16):

                print("testing "+str(neuron_number)+" neurons in the 1 hidden layer.")# 5 neurons fixed on the 2 hidden layer")
                print("------------------------------------------")
                for train, test in kfold.split(X_pca_train, y_train):

                    # 1. define the network
                    #    A network is a sequence of layers in the Sequential class

                    model = Sequential()
                    model.add(Dense(i*l, input_dim=k, activation='tanh')) # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
                    model.add(Dense(neuron_number, activation='tanh'))
                    model.add(Dense(1, activation='sigmoid'))

                    # 2. compile the network
                    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

                    # 3. fit the network
                    history = model.fit(X_pca_train[train], y_train[train], epochs=150, batch_size=10, verbose=0)

                    # 4. evaluate the network
                    scores = model.evaluate(X_pca_train[test], y_train[test], verbose=0)

                    print("rms %.4f%% %s: %.4f%%" % (scores[0] * 100, model.metrics_names[1], scores[1] * 100))
                    cvscores.append(scores[1] * 100)
                    cvmse.append(scores[0] * 100)
                    K.clear_session()

                print("mean\nrms %.4f%% score %.4f%% standard deviation(+/- %.4f%%)" % (numpy.mean(cvmse), numpy.mean(cvscores), numpy.std(cvscores)))
                print("------------------------------------------")

                print("-----------------Final--------------------")





main()
