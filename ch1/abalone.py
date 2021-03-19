import numpy as np
import csv
import time

np.random.seed(time.time())

RND_MEAN = 0
RND_STD = 0.003

learningrate = 0.001


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


def load_abalone_dataset():
    with open('abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)
    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + output_cnt])

    for n, row in enumerate(rows):
        if rows[0] == 'I':
            data[n, 0] = 1
        if rows[0] == 'M':
            data[n, 1] = 1
        if rows[0] == 'F':
            data[n, 2] = 1
        data[n, 3:] = rows[1:]


