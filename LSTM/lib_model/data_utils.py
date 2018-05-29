# coding=utf-8

# Dataset needs to be shuffled before calling TextReader() instance;
# Use shuffle_dataset() to make a new CSV file, default is /datasets/train_set.csv

import codecs
import random, csv
import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import sys
import string
import os

printable = string.printable

# PATH needs to be changed accordingly
PATH = 'LSTM/'
TRAIN_SET = PATH +'Datasets/labeled_data.csv'#'Datasets/All_Tweets_June2016_Dataset.csv'
TEST_SET = PATH + 'Datasets/labeled_data.csv'#'Datasets/labeled_data.csv'
VALID_PERC = 0.05

# TODO: Add non-Ascii characters
emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '


DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
ALPHABET_SIZE = len(emb_alphabet)

def reshape_lines(lines):
    data = []
    for l in lines:
        split = l.split('","')
        data.append((split[0][1:], split[-1][:-2]))
    return data

def save_csv(out_file, data):
    with open(out_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    #print('Data saved to file: %s' % out_file)


TRAIN_SET = PATH + 'datasets/train_set.csv'
TEST_SET = PATH + 'datasets/test_set.csv'
VALID_SET = PATH + 'datasets/valid_set.csv'

class TextReader(object):
    """ Util for Reading the Stanford CSV Files """
    # TODO: Add support for larger files and Queues

    def __init__(self, file, max_word_length):
        # TextReader() takes a CSV file as input that it will read
        # through a buffer

        if file != None:
            self.file = file
        self.max_word_length = max_word_length

    def encode_one_hot(self, sentence):
        # Convert Sentences to np.array of Shape ('sent_length', 'word_length', 'emb_size')

        max_word_length = self.max_word_length
        sent = []
        SENT_LENGTH = 0
        encoded_sentence = "".join(filter(lambda x: x in (printable), sentence))

        print(encoded_sentence)
        for word in word_tokenize(encoded_sentence):

            word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))

            for i, char in enumerate(word):

                try:
                    char_encoding = DICT[char]
                    one_hot = np.zeros(ALPHABET_SIZE)
                    one_hot[char_encoding] = 1
                    word_encoding[i] = one_hot

                except Exception as e:
                    pass

            sent.append(np.array(word_encoding))
            SENT_LENGTH += 1

        return np.array(sent), SENT_LENGTH

    def make_minibatch(self, sentences):
        # Create a minibatch of sentences and convert sentiment
        # to a one-hot vector, also takes care of padding

        max_word_length = self.max_word_length
        minibatch_x = []
        minibatch_y = []
        max_length = 0

        for sentence in sentences:
            # 0: Negative 1: Positive
            minibatch_y.append(np.array([0, 1]) if sentence[:1] == '0' else np.array([1, 0]))
            one_hot, length = self.encode_one_hot(sentence[2:-1])

            if length >= max_length:
                max_length = length
            minibatch_x.append(one_hot)


        # data is a np.array of shape ('b', 's', 'w', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get ('b', 'SENTENCE_MAX_LENGTH', 'WORD_MAX_LENGTH', 'e')
        def numpy_fillna(data):
            # Get lengths of each row of data
            enc = []
            
            #for ent in data:
                #encoded = self.encode_one_hot(ent)
                #enc.append(encoded)
            lens = np.array([len(i) for i in data])

            #print('LENGTH!!!!!!!!!!!!!!')
            #for e in lens:
            #    print(e)
            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            out = np.zeros(shape=(mask.shape + (max_word_length, ALPHABET_SIZE)),
                           dtype='float32')
            #for ent in data:
                #encoded = self.encode_one_hot(ent)
                #enc.append(encoded)

            #out[mask] = np.concatenate(data)
            return out

        # Padding...
        minibatch_x = numpy_fillna(minibatch_x)

        return minibatch_x, np.array(minibatch_y)

    def load_to_ram(self, batch_size):
        # Load n Rows from File f to Ram

        self.data = []
        n_rows = batch_size
        while n_rows > 0:
            self.data.append(next(self.file))
            n_rows -= 1
        if n_rows == 0:
            return True
        else:
            return False

    def iterate_minibatch(self, batch_size, dataset=TRAIN_SET):
        # Returns Next Batch and Catch Bound Errors
        if dataset == TRAIN_SET:
            n_samples = 4800#1600000 * 0.95
        elif dataset == VALID_SET:
            n_samples = 4800#1600000 * 0.05
        else: #elif dataset == TEST_SET:
            n_samples = 4750

        n_batch = int(n_samples // batch_size)

        for i in range(n_batch):
            if self.load_to_ram(batch_size):
                inputs, targets = self.make_minibatch(self.data)
                yield inputs, targets
