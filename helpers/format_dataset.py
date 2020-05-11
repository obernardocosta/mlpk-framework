from csv import reader
import csv
import pandas as pd
import numpy as np

import adjust_data as ad

# Load a csv file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def pop_samples_to_format(dataset, n):
    for i in range(n-1):
        dataset.pop(0)
    return dataset


def get_answer(csv, samples_per_bucket=0):
    formated = []
    for i in range(len(csv)-1):

        if float(csv[i+1][1]) > float(csv[i][1]):
            formated.append(1)
        else:
            formated.append(0)

    if (samples_per_bucket > 0):
        formated = pop_samples_to_format(formated, samples_per_bucket)

    return formated


def get_samples_input(input_csv, samples_per_bucket=1):
    input = []
    input_formated = []

    for sample in input_csv:
        input.append(sample[1])

    line = []
    for i in range(len(input) - samples_per_bucket):
        for j in range(samples_per_bucket):
            line.append(input[j + i])
        input_formated.append(line)
        line = []

    return input_formated


def combine(input, answer):
    dataset = []
    line = []
    for i in range(len(input)):
        line = input[i]
        line.append(str(answer[i]))
        dataset.append(line)
        line = []
    return dataset


def format_dataset(input_csv, output_csv, samples_per_bucket):
    answer = get_answer(output_csv, samples_per_bucket)
    input = get_samples_input(input_csv, samples_per_bucket)
    dataset = combine(input, answer)
    return dataset


def get_dataset(input_csv, output_csv, path=""):
    if path != "":
        path = "-" + path

    for i in range(1, 31):
        print(str(i) + "/30")
        myfile = format_dataset(input_csv, output_csv, i)

        with open("./../../data/dirty"+path+"/data-" + str(i) + ".csv", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in myfile:
                writer.writerow([val])


def main():
    google_trends_filename = './../../data/src-datasets/gt.csv'
    charts_csv_filename = './../../data/src-datasets/price.csv'

    google_trends_csv = load_csv(google_trends_filename)
    charts_csv = (load_csv(charts_csv_filename))

    #get btc only data
    get_dataset(charts_csv, charts_csv)

    #get btc with gtrends data
    get_dataset(google_trends_csv, charts_csv, path="gt")

    ad.adjust_all_datasets()


if __name__ == "__main__":
    main()