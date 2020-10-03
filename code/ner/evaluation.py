#All required packages are loaded
import shutil
import math
import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score
from conlleval import conlleval
import numpy as np
from tensorflow.keras import Model, Input
from datetime import datetime

#In this section the inputs needed by the script are provided

#Provide the working directory, this is the location where this script is located. e.g: /Users/adithya/PycharmProjects/ner_project/code/ner/
working_directory = "/Users/adithya/PycharmProjects/ner_project/code/ner/"
pre_processing_iteration = "execution/iteration0"
priority_array_input = [0.24089703400048226, 0.29920171937365675, 0.4449191900427271, 0.20218214206190158,
                  0.23652694610778444, 0.28697075066475763, 0.34636080430472954, 0.18147625763986835,
                  0.10918054470029459]
worker_annotators_count = 47

#variables that need not be touched
training_sentence_count_prev_iteration = 0
training_sentence_count_this_iteration = 1
NUM_RUNS = 30
EMBEDDING_DIM = 300
BATCH_SIZE = 64


#functions for execution
def no_of_annotations_cnn():
    number_of_annotations = 0
    for i in range(0,len(ystar_x_words)):
        number_of_annotations = number_of_annotations + int(number_of_annotations_sentence[i][0])
    return number_of_annotations
def sum_of_distance_cnn():
    sum_of_distance = 0
    for j in range(0,len(ystar_x_words)):
        distance_for_sentence = 0
        for i in range(0,len(ystar_x_labels[j])):
            distance_for_sentence = distance_for_sentence + distance_between_labels(ystarcap_label_encoding[j][i],ystar_label_aggregate[j][i])
        sum_of_distance = sum_of_distance + (distance_for_sentence*confidence_measurement_data[j])
    return sum_of_distance

def confidence_measurement_word(sentence, word):
    a = sentence
    b = word
    aggregate_prob_array = []
    for j in range(0, len(ystar_label_aggregate[a][b])):
        if ystar_label_aggregate[a][b][j] > 0:
            aggregate_prob_array.append(ystar_label_aggregate[a][b][j])
    minimum = aggregate_prob_array[0]
    maximum = aggregate_prob_array[0]
    if len(aggregate_prob_array) == 1:
        confidence_measure_word = 1
    else:
        for k in range(0, len(aggregate_prob_array)):
            value = aggregate_prob_array[k]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        confidence_measure_word = maximum-minimum
    return confidence_measure_word

def confidence_measurement_word_cnn(sentence, word):
    a = sentence
    b = word
    aggregate_prob_array = []
    for j in range(0, len(cnn_test_ystar_label_aggregate[a][b])):
        if cnn_test_ystar_label_aggregate[a][b][j] > 0:
            aggregate_prob_array.append(cnn_test_ystar_label_aggregate[a][b][j])
    minimum = aggregate_prob_array[0]
    maximum = aggregate_prob_array[0]
    if len(aggregate_prob_array) == 1:
        confidence_measure_word = 1
    else:
        for k in range(0, len(aggregate_prob_array)):
            value = aggregate_prob_array[k]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        confidence_measure_word = maximum-minimum
    return confidence_measure_word

def no_of_annotations_cnn_test():
    number_of_annotations = 0
    for i in range(0,len(cnn_x_words)):
        number_of_annotations = number_of_annotations + int(number_of_annotations_cnn[i])
    return number_of_annotations
def sum_of_distance_cnn_test():
    sum_of_distance = 0
    for j in range(0,len(cnn_x_words)):
        distance_for_sentence = 0
        for i in range(0,len(cnn_x_labels[j])):
            distance_for_sentence = distance_for_sentence + (distance_between_labels(cnn_test_ystarcap_label_encoding[j][i],cnn_test_ystar_label_aggregate[j][i]))
        sum_of_distance = sum_of_distance + (distance_for_sentence*cnn_confidence_measurement[j])
        #sum_of_distance = sum_of_distance + distance_for_sentence
    return sum_of_distance


def eval_model(model):
    pr_test = model.predict(X_test_enc, verbose=2)
    pr_test = np.argmax(pr_test, axis=2)

    yh = y_test_enc.argmax(2)
    fyh, fpr = score(yh, pr_test)
    print("Testing accuracy:" + str(accuracy_score(fyh, fpr)))
    print("Testing confusion matrix:")
    print(confusion_matrix(fyh, fpr))

    preds_test = []
    for i in range(len(pr_test)):
        row = pr_test[i][-len(y_test[i]):]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    preds_test = [list(map(lambda x: ind2label[x], y)) for y in preds_test]
    results_test = conlleval(preds_test, y_test, X_test, 'cnn_predict_output.txt')
    print("Results for testset:" + str(results_test))

    cnn_predict_output_formatted = open("cnn_predict_output_formatted.txt", "a")

    for x in range(0, len(preds_test)):
        for y in range(0, len(preds_test[x])):
            cnn_predict_output_formatted.write(X_test[x][y])
            cnn_predict_output_formatted.write(" ")
            cnn_predict_output_formatted.write(preds_test[x][y])
            cnn_predict_output_formatted.write('\n')
        cnn_predict_output_formatted.write('\n')
    cnn_predict_output_formatted.close()

    return results_test

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

def entropy_calc(first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = weight_of_worker * test * math.log(test,2)
    return entropy_test
def entropy_calc_sentence(first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = test * math.log(test,2)
    return entropy_test

def onehot_vector_encoding(label):
    test = label2ind.get(label, -1)
    if test != -1:
        one_hot_vector = encode(test-1,len(labels))
    else:
        one_hot_vector = np.zeros(len(labels))
    return one_hot_vector

def multiply_by_worker_weight_one(answers):
    multiply_answers = []
    priority_array = priority_array_input
    Labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    for i in range(0,len(answers)):
        if np.array_equal(answers[i],onehot_vector_encoding(Labels[0])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[0]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[1])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[1]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[2])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[2]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[3])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[3]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[4])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[4]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[5])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[5]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[6])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[6]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[7])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[7]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[8])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[8]
        else:
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[8]
        multiply_answers.append(multiply_answers_test)
    return multiply_answers

def ystar_label_one(answers):
    answers_updated = multiply_by_worker_weight_one(answers)
    ystar_label = sum(answers_updated)/np.sum(sum(answers_updated))
    return ystar_label

def multiply_by_worker_weight(answers, ystarcapanswer):
    multiply_answers = []
    priority_array = priority_array_input
    for i in range(0, len(answers)):
        if np.array_equal(answers[i], onehot_vector_encoding(Labels[0])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[0]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[1])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[1]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[2])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[2]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[3])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[3]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[4])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[4]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[5])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[5]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[6])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[6]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[7])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[7]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[8])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[8]
        else:
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[8]
        multiply_answers.append(multiply_answers_test)
    if np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[0])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[0]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[1])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[1]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[2])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[2]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[3])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[3]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[4])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[4]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[5])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[5]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[6])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[6]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[7])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[7]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[8])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[8]
    else:
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[8]
    multiply_answers.append(multiply_answers_test_ystarcap)
    return multiply_answers


def ystar_label(answers, ystarcapanswer):
    answers_updated = multiply_by_worker_weight(answers, ystarcapanswer)
    ystar_label = sum(answers_updated) / numpy.sum(sum(answers_updated))
    return ystar_label


def distance_between_labels(label1, label2):
    for i in range(0, len(label1)):
        if label1[i] > 0:
            ranger = i
    distance = -math.log(label2[ranger], 2)
    return distance

def focal_loss(gamma=5., alpha=2.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def build_base_model():
    model = Embedding(input_dim=num_words,
                      output_dim=300,
                      weights=[embedding_matrix],
                      input_length=maxlen,
                      trainable=True)(input_word)
    model = Dropout(0.5)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(N_CLASSES, activation="softmax"))(model)
    model = Model(input_word, out)
    model.compile(loss=focal_loss(alpha=1), optimizer='nadam', metrics=['accuracy'])
    return model


def no_of_annotations(worker_number):
    a = worker_number
    number_of_annotations = 0
    for i in range(0, len(ystar_x_words)):
        k = 0
        if ystar_x_labels[i][k][a] != '?':
            number_of_annotations = number_of_annotations + int(number_of_annotations_sentence[i][0])
    return number_of_annotations

def sentences_worked_by_worker(worker_number):
    a = worker_number
    sentence_list_worked_by_worker = []
    for i in range(0,len(ystar_x_words)):
        k=0
        if ystar_x_labels[i][k][a] !='?':
            sentence_list_worked_by_worker.append(i)
    return sentence_list_worked_by_worker

def sum_of_distance(worker_number):
    a = worker_number
    sum_of_distance = 0
    x = []
    x = sentences_worked_by_worker(a)
    for m in range(0,len(x)):
        z = x[m]
        distance_for_sentence = 0
        for i in range(0,len(ystar_x_labels[z])):
            distance_for_sentence = distance_for_sentence + (distance_between_labels(ystar_test_label_encoding[z][i][a],ystar_label_aggregate[z][i]))
        sum_of_distance = sum_of_distance + (distance_for_sentence*confidence_measurement_data[z])
    return sum_of_distance
def sum_of_confidence_measure(worker_number):
    a = worker_number
    sum_of_confidence_measure = 0
    x = []
    x = sentences_worked_by_worker(a)
    for m in range(0,len(x)):
        z = x[m]
        sum_of_confidence_measure = sum_of_confidence_measure + confidence_measurement_data[z]
    return sum_of_confidence_measure

span_loss_matrix = []
for i in range(0,9):
    internal_array = []
    for j in range(0,9):
        internal_array.append(0)
    span_loss_matrix.append(internal_array)

print(span_loss_matrix)

for j in range(0,9):
    span_loss_matrix[4][j]=10000
    span_loss_matrix[5][j]=10000
    span_loss_matrix[6][j]=10000
    span_loss_matrix[7][j]=10000
print(span_loss_matrix)

span_loss_matrix[4][0]=0
span_loss_matrix[4][4]=0
span_loss_matrix[5][1]=0
span_loss_matrix[5][5]=0
span_loss_matrix[6][2]=0
span_loss_matrix[6][6]=0
span_loss_matrix[7][3]=0
span_loss_matrix[7][7]=0

print(span_loss_matrix)

def minimum_label(index_array, sentence_number, word_number):
    a = sentence_number
    i = word_number
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    minimum = index_array[0]
    for k in range(0, len(index_array)):
        index = index_array[k]
        label_value = -math.log(ystar_label_aggregate[a][i][index], 2)
        min_value = -math.log(ystar_label_aggregate[a][i][minimum], 2)
        if label_value < min_value:
            minimum = index_array[k]
    min_label_aggregate_one_hot = encode(minimum,len(labels))
    for k in range(0,len(actual_label_one_hot)):
        if np.array_equal(actual_label_one_hot[k],min_label_aggregate_one_hot):
            min_label_aggregate = labels[k]
    return min_label_aggregate, minimum

def span_loss_calculation(label1, label2):
    label1_internal = label1
    label2_internal = label2
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    for i in range(0,len(labels)):
        if label1_internal == labels[i]:
            label1_index = i
        if label2_internal == labels[i]:
            label2_index = i
    span_loss = span_loss_matrix[label2_index][label1_index]
    return span_loss


def confidence_measurement(sentence_number):
    a = sentence_number
    sentence_length = len(ystar_label_aggregate[a])
    confidence_measure_sentence = 0
    for i in range(0, len(ystar_label_aggregate[a])):
        aggregate_prob_array = []
        for j in range(0, len(ystar_label_aggregate[a][i])):
            if ystar_label_aggregate[a][i][j] > 0:
                aggregate_prob_array.append(ystar_label_aggregate[a][i][j])
        minimum = aggregate_prob_array[0]
        maximum = aggregate_prob_array[0]
        if len(aggregate_prob_array) == 1:
            confidence_measure_word = 1
        else:
            for k in range(0, len(aggregate_prob_array)):
                value = aggregate_prob_array[k]
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
            confidence_measure_word = maximum - minimum

        confidence_measure_sentence = confidence_measure_sentence + confidence_measure_word
    confidence_measurement = confidence_measure_sentence / sentence_length
    return confidence_measurement


def gradient_descent_function(sentence_number):
    a = sentence_number
    probability_label = []
    b_labels = ['B-ORG', 'B-LOC', 'B-MISC', 'O', 'B-PER']
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    label_array = actual_label_one_hot
    for i in range(0, len(ystar_label_aggregate[a])):
        probability_array = []
        sentence_length = len(ystar_label_aggregate[a])
        length_prob_array = len(probability_label)
        if length_prob_array == 0:
            index_array = []
            for j in range(0, len(ystar_label_aggregate[a][i])):
                if ystar_label_aggregate[a][i][j] > 0:
                    index_array.append(j)
            min_label_first, minimum = minimum_label(index_array, a, i)
            if min_label_first in b_labels:
                probability_label.append(min_label_first)
            else:
                index_array.remove(minimum)
                if len(index_array) > 0:
                    min_label_second, minimum = minimum_label(index_array, a, i)
                    if min_label_second in b_labels:
                        probability_label.append(min_label_second)
                else:
                    probability_label.append(min_label_first)
        elif length_prob_array >= 1:
            index_array = []
            index_array1 = []
            sentence_length_internal = sentence_length - 1
            if length_prob_array < sentence_length_internal:
                probability_label_first = []
                probability_label_second = []
                for j in range(0, len(ystar_label_aggregate[a][i])):
                    if ystar_label_aggregate[a][i][j] > 0:
                        index_array.append(j)
                min_label_first, minimum = minimum_label(index_array, a, i)
                probability_label_first.append(min_label_first)
                index_array.remove(minimum)
                if len(index_array) > 0:
                    min_label_second, minimum = minimum_label(index_array, a, i)
                    probability_label_second.append(min_label_second)

                for j in range(0, len(ystar_label_aggregate[a][i + 1])):
                    if ystar_label_aggregate[a][i + 1][j] > 0:
                        index_array1.append(j)
                min_label_first, minimum = minimum_label(index_array1, a, i + 1)
                probability_label_first.append(min_label_first)
                index_array1.remove(minimum)
                if len(index_array1) > 0:
                    min_label_second, minimum = minimum_label(index_array1, a, i + 1)
                    probability_label_second.append(min_label_second)

                span_loss_first = span_loss_calculation(probability_label[i - 1],
                                                        probability_label_first[0]) + span_loss_calculation(
                    probability_label_first[0], probability_label_first[1])
                if len(probability_label_second) > 0:
                    span_loss_third = span_loss_calculation(probability_label[i - 1],
                                                            probability_label_second[0]) + span_loss_calculation(
                        probability_label_second[0], probability_label_first[1])
                if len(probability_label_second) > 1:
                    span_loss_second = span_loss_calculation(probability_label[i - 1],
                                                             probability_label_first[0]) + span_loss_calculation(
                        probability_label_first[0], probability_label_second[1])
                    span_loss_fourth = span_loss_calculation(probability_label[i - 1],
                                                             probability_label_second[0]) + span_loss_calculation(
                        probability_label_second[0], probability_label_second[1])

                if span_loss_first == 0:
                    probability_label.append(probability_label_first[0])
                elif len(probability_label_second) > 1 and span_loss_second == 0:
                    probability_label.append(probability_label_first[0])
                elif len(probability_label_second) > 0 and span_loss_third == 0:
                    probability_label.append(probability_label_second[0])
                elif len(probability_label_second) > 1 and span_loss_fourth == 0:
                    probability_label.append(probability_label_second[0])
                else:
                    probability_label.append(probability_label_first[0])
            elif length_prob_array == sentence_length_internal:
                index_array = []
                for j in range(0, len(ystar_label_aggregate[a][i])):
                    if ystar_label_aggregate[a][i][j] > 0:
                        index_array.append(j)
                min_label_first, minimum = minimum_label(index_array, a, i)
                index_array.remove(minimum)
                if len(index_array) > 0:
                    min_label_second, minimum = minimum_label(index_array, a, i)
                span_loss_first = span_loss_calculation(probability_label[i - 1], min_label_first)
                if len(index_array) > 0:
                    span_loss_second = span_loss_calculation(probability_label[i - 1], min_label_second)
                if span_loss_first == 0:
                    probability_label.append(min_label_first)
                elif len(index_array) > 0 and span_loss_second == 0:
                    probability_label.append(min_label_second)
                else:
                    probability_label.append(min_label_first)

    return probability_label


#Execution begins from here
datetimeobj = datetime.now()
date = datetimeobj.strftime("%d_%b_%y")
execution_dir = "execution_" + date
iteration = 0
while(training_sentence_count_prev_iteration!=training_sentence_count_this_iteration or training_sentence_count_this_iteration<training_sentence_count_prev_iteration):
    training_sentence_count_prev_iteration = training_sentence_count_this_iteration
    iteration = iteration + 1
    iteration_dir = "iteration" + str(iteration) + '/'
    prev_iteration_dir = "iteration" + str(iteration-1) + '/'
    #Change to the working directory and create necessary folder structures
    os.chdir(working_directory)
    path = os.getcwd()
    print(path)
    os.system('mkdir -p ' + execution_dir + '/' + iteration_dir)
    os.chdir(execution_dir + '/' + iteration_dir)
    path_new = os.getcwd()
    print(path_new)
    DATA_PATH = working_directory + execution_dir + '/' + iteration_dir + '/'
    #Create necessary folder structure for pre-processing
    os.system('mkdir -p data')
    os.system('mkdir -p dictionary_directory/dictionary_sentence_list')
    os.system('mkdir -p dictionary_directory/dictionary_aggregate_sentence_list')
    os.system('mkdir -p dictionary_directory/dictionary_gt_sentence_list')
    os.system('mkdir individual_files')
    os.system('mkdir -p dictionary_directory/dictionary_worker_sentence')
    os.system('mkdir -p dictionary_directory/dictionary_sentence_worker')
    os.system('mkdir -p dictionary_directory/dictionary_aggregate_'+ iteration_dir + '_sentence_list')


    # Copy required files for processing
    conllevalpl = shutil.copy("../../../../conlleval.pl", "conlleval.pl")
    conllevalpy = shutil.copy("../../../../conlleval.py", "conlleval.py")
    conllevalpyc = shutil.copy("../../../../conlleval.pyc", "conlleval.pyc")
    answers = shutil.copy("../../../../datasets/ner/answers.txt", "data/answers.txt")
    groundtruth = shutil.copy("../../../../datasets/ner/Ground_Truth.txt", "data/Ground_Truth.txt")
    number_of_annotations = shutil.copy("../../" + pre_processing_iteration + "/individual_files/number_of_annotations.txt", "data/number_of_annotations.txt")
    word_index = shutil.copy("../../" + pre_processing_iteration + "/individual_files/word_index.txt", "data/word_index.txt")
    dataset_answers = shutil.copy("../../../../datasets/ner/answers.txt", "dataset_sentence_complete.txt")
    dataset_ground_truth = shutil.copy("../../../../datasets/ner/Ground_Truth.txt", "dataset_sentence_gt.txt")

    if(iteration==1):
        aggregate_label = shutil.copy("../../../../datasets/ner/mv.txt", "data/aggregation.txt")
        dataset_aggregate_label = shutil.copy("../../../../datasets/ner/mv.txt", "dataset_sentences_aggregate.txt")
        worker_weight = shutil.copy("../../" + pre_processing_iteration + "/data/worker_weight.txt", "data/worker_weight.txt")
        worker_weight_cnn = shutil.copy("../../" + pre_processing_iteration + "/data/r_theta.txt", "data/r_theta.txt")
    else:
        aggregate_label = shutil.copy("../" + prev_iteration_dir + "/total_sentences_aggregate_cnn.txt", "data/aggregation.txt")
        dataset_aggregate_label = shutil.copy("../" + prev_iteration_dir + "/total_sentences_aggregate_cnn.txt", "dataset_sentences_aggregate.txt")
        worker_weight = shutil.copy("../" + prev_iteration_dir + "/updated_worker_weight.txt", "data/worker_weight.txt")
        worker_weight_cnn = shutil.copy("../" + prev_iteration_dir + "/updated_rtheta.txt", "data/r_theta.txt")


    #Load worker_weights and cnn_weights

    worker_weights = []
    worker_weight_file = open("data/worker_weight.txt", "r")

    for line in worker_weight_file:
        line1 = line.rstrip()
        worker_weights.append(float(line1))
    worker_weight_file.close()
    print(worker_weights)
    print(len(worker_weights))

    worker_weight_cnn = []
    worker_weight_cnn_file = open("data/r_theta.txt", "r")
    for line in worker_weight_cnn_file:
        line1 = line.rstrip()
        worker_weight_cnn.append(float(line1))
    worker_weight_cnn_file.close()
    print(worker_weight_cnn)
    print(len(worker_weight_cnn))

    #Load number of annotations to consider
    annotations = read_conll('data/number_of_annotations.txt')
    number_of_annotations_sentence = [[c[1] for c in x] for x in annotations]
    print(int(number_of_annotations_sentence[0][0]))

    all_answers = read_conll(DATA_PATH + 'dataset_sentence_complete.txt')
    all_mv = read_conll(DATA_PATH + 'dataset_sentences_aggregate.txt')
    all_test = read_conll(DATA_PATH + 'dataset_sentences_aggregate.txt')
    all_docs = all_test
    length_all_answers = len(all_answers)
    length_all_mv = len(all_mv)
    length_all_test = len(all_test)
    length_all_docs = len(all_docs)
    print("Answers data size:" + str(length_all_answers))
    print("Majority voting data size:" + str(length_all_mv))
    print("Test data size:" + str(length_all_test))
    print("Total sequences:" + str(length_all_docs))

    X_train = [[c[0] for c in x] for x in all_answers]
    y_answers = [[c[1:] for c in y] for y in all_answers]
    y_mv = [[c[1] for c in y] for y in all_mv]
    X_test = [[c[0] for c in x] for x in all_test]
    y_test = [[c[1] for c in y] for y in all_test]
    X_all = [[c[0] for c in x] for x in all_docs]
    y_all = [[c[1] for c in y] for y in all_docs]

    N_ANNOT = len(y_answers[0][0])
    print("Num annnotators:" + str(N_ANNOT))

    lengths = [len(x) for x in all_docs]
    all_text = [c for x in X_all for c in x]
    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    print("Labels:" + str(labels))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    ind2label[0] = "O"  # padding index
    max_length = max(lengths)
    min_length = min(lengths)
    print("Input sequence length range: " + " Max " + str(max_length) + " Min " + str(min_length))

    max_label = max(label2ind.values()) + 1
    print("Max label:" + str(max_label))

    maxlen = max([len(x) for x in X_all])
    print("Maximum sequence length:" + str(maxlen))

    Labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    labels_one_hot = []
    for i in range(0, len(Labels)):
        label_one_hot = onehot_vector_encoding(Labels[i])
        labels_one_hot.append(label_one_hot)
    print(labels_one_hot)

    actual_label_one_hot = []
    for i in range(len(labels)):
        actual_label_one_hot_i = onehot_vector_encoding(labels[i])
        actual_label_one_hot.append(actual_label_one_hot_i)
    print(actual_label_one_hot)

    ystar_x_answers = read_conll(DATA_PATH + 'dataset_sentence_complete.txt')
    ystar_x_words = [[c[0] for c in x] for x in ystar_x_answers]
    ystar_x_labels = [[c[1:] for c in y] for y in ystar_x_answers]
    print(ystar_x_labels[0][0])

    if (iteration == 1):
        ystar_test_label_encoding = []
        for i in range(0, len(ystar_x_labels)):
            ystar_test_label_i = []
            for j in range(0, len(ystar_x_labels[i])):
                ystar_test_label_j = []
                for k in range(0, len(ystar_x_labels[i][j])):
                    label_y_one_hot = onehot_vector_encoding(ystar_x_labels[i][j][k])
                    ystar_test_label_j.append(label_y_one_hot)
                ystar_test_label_i.append(ystar_test_label_j)
            ystar_test_label_encoding.append(ystar_test_label_i)

        print(ystar_test_label_encoding[0][0])

    ystar_label_aggregate = []
    for i in range(0, len(ystar_test_label_encoding)):
        ystar_label_aggregate_i = []
        for j in range(0, len(ystar_test_label_encoding[i])):
            label_ystar_aggregate = ystar_label_one(ystar_test_label_encoding[i][j])
            ystar_label_aggregate_i.append(label_ystar_aggregate)
        ystar_label_aggregate.append(ystar_label_aggregate_i)
    print(ystar_label_aggregate[0])

    confidence_measurement_data = []
    for i in range(0, len(ystar_label_aggregate)):
        confidence_measurement_data.append(confidence_measurement(i))
    print(len(confidence_measurement_data))
    print(confidence_measurement_data[0])

    test_sentences = []
    train_sentences = []
    for i in range(0, len(confidence_measurement_data)):
        if confidence_measurement_data[i] <= 0.9:
            test_sentences.append(i)
        elif confidence_measurement_data[i] > 0.9:
            train_sentences.append(i)
    print(len(test_sentences))
    print(len(train_sentences))
    print(len(test_sentences) + len(train_sentences))

    training_sentence_count_this_iteration = len(train_sentences)

    # Finding updated worker weights for all workers
    lambda_1 = 1
    for i in range(0, worker_annotators_count):
        sentences_by_worker = sentences_worked_by_worker(i)
        if len(sentences_by_worker) != 0:
            weight_of_worker = (lambda_1 * no_of_annotations(i)) / sum_of_distance(i)
        worker_weights[i] = weight_of_worker

    print(worker_weights)

    # Finding aggregate labels for all the sentences
    Label_sentence_agg = []
    for i in range(0, len(ystar_x_words)):
        Label_sentence = []
        Label_sentence = gradient_descent_function(i)
        Label_sentence_agg.append(Label_sentence)
    print(Label_sentence_agg)

    Label_aggregate_testing = []
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        Label_aggregate_testing.append(Label_sentence_agg[sentence_number])

    Label_aggregate_train = []
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        Label_aggregate_train.append(Label_sentence_agg[sentence_number])


    total_sentences_aggregate = open("total_sentences_aggregate.txt", "a")
    for i in range(0, len(Label_sentence_agg)):
        for j in range(0, len(Label_sentence_agg[i])):
            total_sentences_aggregate.write(str(ystar_x_words[i][j]))
            total_sentences_aggregate.write(" ")
            total_sentences_aggregate.write(str(Label_sentence_agg[i][j]))
            total_sentences_aggregate.write("\n")
        total_sentences_aggregate.write("\n")
    total_sentences_aggregate.close()

    train_sentences_aggregate = open("train_sentences_aggregate.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_aggregate.write(" ")
            train_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            train_sentences_aggregate.write("\n")
        train_sentences_aggregate.write("\n")
    train_sentences_aggregate.close()

    train_sentences_complete = open("train_sentences_complete.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_complete.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_complete.write(" ")
            for k in range(0, len(ystar_x_labels[sentence_number][j])):
                train_sentences_complete.write(str(ystar_x_labels[sentence_number][j][k]))
                train_sentences_complete.write(" ")
            train_sentences_complete.write("\n")
        train_sentences_complete.write("\n")
    train_sentences_complete.close()

    test_sentences_aggregate = open("test_sentences_aggregate.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            test_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            test_sentences_aggregate.write(" ")
            test_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            test_sentences_aggregate.write("\n")
        test_sentences_aggregate.write("\n")
    test_sentences_aggregate.close()

    confidence_measure_file = open("confidence_measure_file.txt", "a")
    for i in range(0, len(confidence_measurement_data)):
        confidence_measure_file.write(str(confidence_measurement_data[i]))
        confidence_measure_file.write("\n")
    confidence_measure_file.close()

    gt_answers = read_conll(DATA_PATH + 'dataset_sentence_gt.txt')
    gt_words = [[c[0] for c in x] for x in gt_answers]
    gt_labels = [[c[1] for c in y] for y in gt_answers]
    print(ystar_x_labels[0][0])

    train_sentences_groundtruth = open("train_sentences_groundtruth.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(gt_words[sentence_number])):
            train_sentences_groundtruth.write(str(gt_words[sentence_number][j]))
            train_sentences_groundtruth.write(" ")
            train_sentences_groundtruth.write(str(gt_labels[sentence_number][j]))
            train_sentences_groundtruth.write("\n")
        train_sentences_groundtruth.write("\n")
    train_sentences_groundtruth.close()

    test_sentences_groundtruth = open("test_sentences_groundtruth.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(gt_words[sentence_number])):
            test_sentences_groundtruth.write(str(gt_words[sentence_number][j]))
            test_sentences_groundtruth.write(" ")
            test_sentences_groundtruth.write(str(gt_labels[sentence_number][j]))
            test_sentences_groundtruth.write("\n")
        test_sentences_groundtruth.write("\n")
    test_sentences_groundtruth.close()

    # Word embeddings
    embeddings_index = {}
    f = open("../../../../glove.6B/glove.6B.300d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors' % len(embeddings_index))

    all_answers = read_conll(DATA_PATH + 'train_sentences_complete.txt')
    all_mv = read_conll(DATA_PATH + 'train_sentences_aggregate.txt')
    all_test = read_conll(DATA_PATH + 'test_sentences_aggregate.txt')
    all_docs = all_mv + all_test
    length_all_answers = len(all_answers)
    length_all_mv = len(all_mv)
    length_all_test = len(all_test)
    length_all_docs = len(all_docs)
    print("Answers data size:" + str(length_all_answers))
    print("Majority voting data size:" + str(length_all_mv))
    print("Test data size:" + str(length_all_test))
    print("Total sequences:" + str(length_all_docs))

    X_train = [[c[0] for c in x] for x in all_answers]
    y_answers = [[c[1:] for c in y] for y in all_answers]
    y_mv = [[c[1] for c in y] for y in all_mv]
    X_test = [[c[0] for c in x] for x in all_test]
    y_test = [[c[1] for c in y] for y in all_test]
    X_all = [[c[0] for c in x] for x in all_docs]
    y_all = [[c[1] for c in y] for y in all_docs]

    N_ANNOT = len(y_answers[0][0])
    print("Num annnotators:" + str(N_ANNOT))

    lengths = [len(x) for x in all_docs]
    all_text = [c for x in X_all for c in x]
    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    print("Labels:" + str(labels))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    ind2label[0] = "O"  # padding index
    max_length = max(lengths)
    min_length = min(lengths)
    print("Input sequence length range: " + " Max " + str(max_length) + " Min " + str(min_length))

    max_label = max(label2ind.values()) + 1
    print("Max label:" + str(max_label))

    maxlen = max([len(x) for x in X_all])
    print("Maximum sequence length:" + str(maxlen))

    num_words = len(word2ind)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2ind.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    X_train_enc = [[word2ind[c] for c in x] for x in X_train]
    y_mv_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_mv]
    y_mv_enc = [[encode(c, max_label) for c in ey] for ey in y_mv_enc]

    y_answers_enc = []
    for r in range(N_ANNOT):
        annot_answers = []
        for i in range(len(y_answers)):
            seq = []
            for j in range(len(y_answers[i])):
                enc = -1
                if y_answers[i][j][r] != "?":
                    enc = label2ind[y_answers[i][j][r]]
                seq.append(enc)
            annot_answers.append(seq)
        y_answers_enc.append(annot_answers)

    X_test_enc = [[word2ind[c] for c in x] for x in X_test]
    y_test_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test]
    y_test_enc = [[encode(c, max_label) for c in ey] for ey in y_test_enc]

    # pad sequences
    X_train_enc = pad_sequences(X_train_enc, maxlen=maxlen)
    y_mv_enc = pad_sequences(y_mv_enc, maxlen=maxlen)
    X_test_enc = pad_sequences(X_test_enc, maxlen=maxlen)
    y_test_enc = pad_sequences(y_test_enc, maxlen=maxlen)

    y_answers_enc_padded = []
    for r in range(N_ANNOT):
        padded_answers = pad_sequences(y_answers_enc[r], maxlen=maxlen)
        y_answers_enc_padded.append(padded_answers)

    y_answers_enc_padded = np.array(y_answers_enc_padded)
    y_answers_enc = np.transpose(np.array(y_answers_enc_padded), [1, 2, 0])

    n_train = len(X_train_enc)
    n_test = len(X_test_enc)

    print("Training and testing tensor shapes:")
    print("X_train_enc.shape " + str(X_train_enc.shape) + "\nX_test_enc.shape " + str(X_test_enc.shape))
    print("y_test_enc.shape" + str(y_test_enc.shape))

    print("Answers shape:" + str(y_answers_enc.shape))

    N_CLASSES = len(label2ind) + 1
    print("Num classes:" + str(N_CLASSES))


    input_word = Input(shape=(60,))
    model = build_base_model()
    model.fit(X_train_enc, y_mv_enc, batch_size=BATCH_SIZE, epochs=20, verbose=2)

    results_test = eval_model(model)

    sentence_name_testing = []

    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_name_testing.append(sentence_name)
    print(len(sentence_name_testing))

    sentence_name_training = []

    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_name_training.append(sentence_name)
    print(len(sentence_name_training))

    sentence_list = []
    for i in range(0, len(Label_sentence_agg)):
        sentence_number = i
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_list.append(sentence_name)
    print(len(sentence_list))

    aggregate_label = open("train_sentences_aggregate.txt", "r")
    counter = 0
    for line in aggregate_label:
        if line not in ['\n', '\r\n']:
            sentence_count = open('dictionary_directory/dictionary_aggregate_'+ iteration_dir + '_sentence_list/' + str(sentence_name_training[counter]), "a")
            sentence_count.write(line)
            sentence_count.close()
        if line in ['\n', '\r\n']:
            counter = counter + 1
    aggregate_label.close()

    aggregate_label = open("cnn_predict_output_formatted.txt", "r")
    counter = 0
    for line in aggregate_label:
        if line not in ['\n', '\r\n']:
            sentence_count = open("dictionary_directory/dictionary_aggregate_"+ iteration_dir + "_sentence_list/" + str(sentence_name_testing[counter]), "a")
            sentence_count.write(line)
            sentence_count.close()
        if line in ['\n', '\r\n']:
            counter = counter + 1
    aggregate_label.close()

    ystar_aggregate_sentence_complete = open("ystarcap_aggregate.txt", "a")
    for i in range(0, len(sentence_list)):
        sentence_file = sentence_list[i]
        sentence_loop = open("dictionary_directory/dictionary_aggregate_"+ iteration_dir + "_sentence_list/" + str(sentence_file) + "", "r")
        for line1 in sentence_loop:
            ystar_aggregate_sentence_complete.write(str(line1))
        ystar_aggregate_sentence_complete.write("\n")
        sentence_loop.close()
    ystar_aggregate_sentence_complete.close()

    ystarcap_prediction = read_conll('ystarcap_aggregate.txt')
    length_ystarcap_prediction = len(ystarcap_prediction)
    print("Length of all prediction: ", length_ystarcap_prediction)
    words_ystarcap = [[c[0] for c in x] for x in ystarcap_prediction]
    ystarcap = [[c[1] for c in x] for x in ystarcap_prediction]
    print(str(words_ystarcap[0]))
    print(str(ystarcap[0]))

    ystarcap_label_encoding = []
    for i in range(0, len(ystarcap)):
        ystarcap_test_label_i = []
        for j in range(0, len(ystarcap[i])):
            label_ystarcap_one_hot = onehot_vector_encoding(ystarcap[i][j])
            ystarcap_test_label_i.append(label_ystarcap_one_hot)
        ystarcap_label_encoding.append(ystarcap_test_label_i)
    print(ystarcap_label_encoding[0])

    actual_label_one_hot = []
    for i in range(len(labels)):
        actual_label_one_hot_i = onehot_vector_encoding(labels[i])
        actual_label_one_hot.append(actual_label_one_hot_i)
    print(actual_label_one_hot)

    import numpy

    ystar_label_aggregate = []
    for i in range(0, len(ystar_test_label_encoding)):
        ystar_label_aggregate_i = []
        for j in range(0, len(ystar_test_label_encoding[i])):
            label_ystar_aggregate = ystar_label(ystar_test_label_encoding[i][j], ystarcap_label_encoding[i][j])
            ystar_label_aggregate_i.append(label_ystar_aggregate)
        ystar_label_aggregate.append(ystar_label_aggregate_i)
    print(ystar_label_aggregate[0])

    # Finding updated worker weights for all workers
    lambda_1 = 1
    for i in range(0, worker_annotators_count):
        sentences_by_worker = sentences_worked_by_worker(i)
        if len(sentences_by_worker) != 0:
            weight_of_worker = (lambda_1 * no_of_annotations(i)) / sum_of_distance(i)
        worker_weights[i] = weight_of_worker

    print(worker_weights)

    cnn_x_answers = read_conll(DATA_PATH + 'cnn_predict_output_formatted.txt')
    cnn_x_words = [[c[0] for c in x] for x in cnn_x_answers]
    cnn_x_labels = [[c[1] for c in y] for y in cnn_x_answers]
    print(cnn_x_labels[0][0])

    cnn_test_ystar_label_aggregate = []
    cnn_test_ystarcap_label_encoding = []
    cnn_confidence_measurement = []
    number_of_annotations_cnn = []
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        cnn_confidence_measurement.append(confidence_measurement_data[sentence_number])
        number_of_annotations_cnn.append(int(number_of_annotations_sentence[sentence_number][0]))
        cnn_test_ystar_label_encoding_internal = []
        cnn_test_ystarcap_label_encoding_internal = []
        for j in range(0, len(ystar_label_aggregate[sentence_number])):
            cnn_test_ystar_label_encoding_internal.append(ystar_label_aggregate[sentence_number][j])
            cnn_test_ystarcap_label_encoding_internal.append(ystarcap_label_encoding[sentence_number][j])
        cnn_test_ystar_label_aggregate.append(cnn_test_ystar_label_encoding_internal)
        cnn_test_ystarcap_label_encoding.append(cnn_test_ystarcap_label_encoding_internal)
    print(cnn_test_ystar_label_aggregate[0])
    print(cnn_test_ystarcap_label_encoding[0])
    print(number_of_annotations_cnn[0])

    weight_of_worker_cnn = (lambda_1 * no_of_annotations_cnn_test()) / sum_of_distance_cnn_test()
    worker_weight_cnn[0] = weight_of_worker_cnn
    print(weight_of_worker_cnn)

    Label_sentence_agg = []
    for i in range(0, len(ystar_x_words)):
        Label_sentence = []
        Label_sentence = gradient_descent_function(i)
        Label_sentence_agg.append(Label_sentence)
    print(Label_sentence_agg)

    updated_worker_weight = open("updated_worker_weight.txt", "a")
    for i in range(0, worker_annotators_count):
        worker_weight = worker_weights[i]
        updated_worker_weight.write(str(worker_weight))
        updated_worker_weight.write("\n")
    updated_worker_weight.close()

    updated_worker_weight_cnn = open("updated_rtheta.txt", "a")
    worker_weight_cnn = worker_weight_cnn[0]
    updated_worker_weight_cnn.write(str(worker_weight_cnn))
    updated_worker_weight_cnn.close()

    total_sentences_aggregate = open("total_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(Label_sentence_agg)):
        for j in range(0, len(Label_sentence_agg[i])):
            total_sentences_aggregate.write(str(ystar_x_words[i][j]))
            total_sentences_aggregate.write(" ")
            total_sentences_aggregate.write(str(Label_sentence_agg[i][j]))
            total_sentences_aggregate.write("\n")
        total_sentences_aggregate.write("\n")
    total_sentences_aggregate.close()

    train_sentences_aggregate = open("train_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_aggregate.write(" ")
            train_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            train_sentences_aggregate.write("\n")
        train_sentences_aggregate.write("\n")
    train_sentences_aggregate.close()

    test_sentences_aggregate = open("test_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            test_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            test_sentences_aggregate.write(" ")
            test_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            test_sentences_aggregate.write("\n")
        test_sentences_aggregate.write("\n")
    test_sentences_aggregate.close()

    word_index = open("data/word_index.txt", "r")
    aggregated_file_updated = open("aggregated_file_updated.txt", "a")
    for line in word_index:
        if line not in ["\n", "\r\n"]:
            index = line.rstrip()
            index_numbers = index.split()
            sentence_number = int(index_numbers[0])
            word_number = int(index_numbers[1])
            aggregated_file_updated.write(str(ystar_x_words[sentence_number][word_number]))
            aggregated_file_updated.write(" ")
            aggregated_file_updated.write(str(Label_sentence_agg[sentence_number][word_number]))
            aggregated_file_updated.write("\n")
        if line in ["\n", "\r\n"]:
            aggregated_file_updated.write("\n")

    aggregated_file_updated.close()
    word_index.close()

    word_index = open("data/word_index.txt", "r")
    aggregated_file_updated = open("gt_file_updated.txt", "a")
    for line in word_index:
        if line not in ["\n", "\r\n"]:
            index = line.rstrip()
            index_numbers = index.split()
            sentence_number = int(index_numbers[0])
            word_number = int(index_numbers[1])
            aggregated_file_updated.write(str(ystar_x_words[sentence_number][word_number]))
            aggregated_file_updated.write(" ")
            aggregated_file_updated.write(str(gt_labels[sentence_number][word_number]))
            aggregated_file_updated.write("\n")
        if line in ["\n", "\r\n"]:
            aggregated_file_updated.write("\n")

    aggregated_file_updated.close()
    word_index.close()































