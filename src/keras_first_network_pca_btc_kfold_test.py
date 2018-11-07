import numpy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sys

from csv import reader
import csv

#gimport matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = None
numpy.random.seed()

#normalizing data for preprocessing
def mean_std(x, mean, std):
    return float((x-mean)/std)

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


def monetary_test(prices, predict, money, order_tax, cashout_tax):


    btc = -9999
    buy = True
    btcs = []
    moneys = []

    print(len(prices))
    print(len(predict))
    for i in range(len(prices)):

        print(prices[i])

        if predict[i] == 1:
            if buy:
                btc = float((money-(money*order_tax))/prices[i])
                money = 0
                buy = False

        elif predict[i] == 0:
            if not buy:
                money = float((btc*prices[i]) - (btc*prices[i]*order_tax))
                btc = 0
                buy = True

        print(money, btc)
        moneys.append(money)
        btcs.append(btc)

    if btc != 0:
        money = float((btc * prices[i]) - (btc * prices[i] * order_tax))
        money -= float((money * cashout_tax))
        btc = 0
        print("final")
        print("money, btc")
        print(money, btc)

    #x = range(len(prices))
    #plt.figure(1)
    #plt.plot(x, prices)
    #plt.ylabel('prices')

    #plt.figure(2)
    #plt.plot(x, moneys)
    #plt.ylabel('money')

    #plt.figure(3)
    #plt.plot(x, btcs)
    #plt.ylabel('btc')

    #plt.show()

def main():

    #
    #TODO - smart implementation for ensemble model
    # this is just a naive implementation.
    #

    data_type = sys.argv[1]
    pca_n = int(sys.argv[2])
    neuron_number = int(sys.argv[3])
    data_n = int(sys.argv[4])
    s = []
    stds = []
    #for steps in range(7):

    k = pca_n
    l = 1

    final = []

    for _ in range(7):
        print("pca with n=", k)
        number_of_folds = k

        cvscores = []

        # load PCA
        pca = PCA(n_components=number_of_folds)

        ## load and prepare the dataset

        path = "./../data/" + data_type + "/data-" + str(data_n) + ".csv"

        factor = 1
        if data_type == "final-mix":
            factor = 2

        dataset = numpy.loadtxt(path, delimiter=",")
        X = dataset[:, 0:factor*data_n * l]
        y = dataset[:, factor*data_n * l]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        X_test_price = X_test

        X_train, X_test = mean_std_transform(X_train, X_test)
        X_pca_train, pca_std = pca_train_data_generator(X_train, pca)
        X_pca_test, pca_std = pca_train_data_generator(X_test, pca)


        #print("testing "+str(neuron_number)+" neurons in the 1 hidden layer.")# 5 neurons fixed on the 2 hidden layer")


        # 1. define the network
        #    A network is a sequence of layers in the Sequential class
        print("11")
        model1 = Sequential()
        model1.add(Dense(data_n*l, input_dim=k, activation='tanh')) # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model1.add(Dense(neuron_number, activation='tanh'))
        model1.add(Dense(1, activation='sigmoid'))

        print("12")
        model2 = Sequential()
        model2.add(Dense(data_n * l, input_dim=k,activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model2.add(Dense(neuron_number, activation='tanh'))
        model2.add(Dense(1, activation='sigmoid'))

        print("13")
        model3 = Sequential()
        model3.add(Dense(data_n * l, input_dim=k, activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model3.add(Dense(neuron_number, activation='tanh'))
        model3.add(Dense(1, activation='sigmoid'))

        print("14")
        model4 = Sequential()
        model4.add(Dense(data_n * l, input_dim=k, activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model4.add(Dense(neuron_number, activation='tanh'))
        model4.add(Dense(1, activation='sigmoid'))

        print("15")
        model5 = Sequential()
        model5.add(Dense(data_n * l, input_dim=k, activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model5.add(Dense(neuron_number, activation='tanh'))
        model5.add(Dense(1, activation='sigmoid'))

        print("16")
        model6 = Sequential()
        model6.add(Dense(data_n * l, input_dim=k, activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model6.add(Dense(neuron_number, activation='tanh'))
        model6.add(Dense(1, activation='sigmoid'))

        print("17")
        model7 = Sequential()
        model7.add(Dense(data_n * l, input_dim=k, activation='tanh'))  # First layer we have Dense(#neurons_hidden layer, input_dim, activation_function)
        model7.add(Dense(neuron_number, activation='tanh'))
        model7.add(Dense(1, activation='sigmoid'))

        # 2. compile the network
        print("21")
        model1.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("22")
        model2.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("23")
        model3.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("24")
        model4.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("25")
        model5.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("26")
        model6.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("27")
        model7.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # 3. fit the network

        print("31")
        history1 = model1.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("32")
        history2 = model2.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("33")
        history3 = model3.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("34")
        history4 = model4.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("35")
        history5 = model5.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("36")
        history6 = model6.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)
        print("37")
        history7 = model7.fit(X_pca_train, y_train, epochs=150, batch_size=10, verbose=0)


        #4. make predictions
        print("41")
        probabilities1 = model1.predict(X_pca_test)
        predictions1 = [float(numpy.round(x)) for x in probabilities1]
        print("42")
        probabilities2 = model2.predict(X_pca_test)
        predictions2 = [float(numpy.round(x)) for x in probabilities2]
        print("43")
        probabilities3 = model3.predict(X_pca_test)
        predictions3 = [float(numpy.round(x)) for x in probabilities3]
        print("44")
        probabilities4 = model4.predict(X_pca_test)
        predictions4 = [float(numpy.round(x)) for x in probabilities4]
        print("45")
        probabilities5 = model5.predict(X_pca_test)
        predictions5 = [float(numpy.round(x)) for x in probabilities5]
        print("46")
        probabilities6 = model6.predict(X_pca_test)
        predictions6 = [float(numpy.round(x)) for x in probabilities6]
        print("47")
        probabilities7 = model7.predict(X_pca_test)
        predictions7 = [float(numpy.round(x)) for x in probabilities7]

        final_predict = []
        for pred in range(len(predictions1)):
            p = []
            p.append(predictions1[pred])
            p.append(predictions2[pred])
            p.append(predictions3[pred])
            p.append(predictions4[pred])
            p.append(predictions5[pred])
            p.append(predictions6[pred])
            p.append(predictions7[pred])
            final_predict.append(numpy.median(p))


        accuracy = numpy.mean(final_predict == y_test)
        print("Prediction Accuracy: %.4f%%" % (accuracy * 100))
        final.append(accuracy * 100)
        print("------------------------------------------")

        #prices = []

        #for line in X_test_price:

        #    btc_prices = line[6]

        #    prices.append(float(btc_prices))

        #print(prices)
        #monetary_test(prices, final_predict, 10000, 0.005, 0.0139)

    print(final)
    print(numpy.median(final))

for _ in range(20):
    main()
