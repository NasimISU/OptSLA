import math
import numpy as np

def entropy_calc(weight_of_worker, first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = weight_of_worker * test * math.log(test,2)
    return entropy_test
def entropy_calc_sentence(first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = test * math.log(test,2)
    return entropy_test

def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x
def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr