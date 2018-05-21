# Some packages we will need:
import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import string
import os

# Our alphabet
emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-' \
               ',;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '

# we associate every character in our alphabet to a number: 
# e.g. b => 1 d => 3 etc.
DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}

# The size of our alphabet (~70)
ALPHABET_SIZE = len(emb_alphabet)
TRAIN_SET ='Datasets/All_Tweets_June2016_Dataset.csv'
TEST_SET ='Datasets/labeled_data.csv'
VALID_SET ='Datasets/labeled_data.csv'

class TextReader(object):
    """ Util for Reading the Stanford CSV Files """

    def __init__(self, file, max_word_length):
        # TextReader() takes a CSV file as input that it will read
        # through a buffer
        
        # we can also feed TextReader() our own sentences, therefore
        # sometimes it will not need a file
        if file != None:
            self.file = file
            
        # The maximum number of character in a word, default is 16 (we
        # will get to this later)
        self.max_word_length = max_word_length

    def encode_one_hot(self, sentence):
        # Convert Sentences to np.array of Shape 
        # ('sent_length', 'word_length', 'emb_size')

        max_word_length = self.max_word_length
        sent = []
        
        # We need to keep track of the maximum length of the sentence in a minibatch
        # so that we can pad them with zeros, this is why we return the length of every
        # sentences after they are converted to one-hot tensors
        SENT_LENGTH = 0
        
        # Here, we remove any non-printable characters in a sentence (mostly
        # non-ASCII characters)
        printable = string.printable
        encoded_sentence = str(filter(lambda x: x in printable, sentence))
        
        # word_tokenize() splits a sentence into an array where each element is
        # a word in the sentence, for example, 
        # "My name is Charles" => ["My", "name", "is", Charles"]
        # Unidecode convert characters to utf-8
        for word in word_tokenize(unidecode(encoded_sentence)):
            
            # Encode one word as a matrix of shape [max_word_length x ALPHABET_SIZE]
            word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))
            
            for i, char in enumerate(word):
            
                # If the character is not in the alphabet, ignore it    
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
            # Append the one-hot encoding of the sentiment to the minibatch of Y
            # 0: Negative 1: Positive
            minibatch_y.append(np.array([0, 1]) if sentence[:1] == '0' else np.array([1, 0]))

            # One-hot encoding of the sentence
            one_hot, length = self.encode_one_hot(sentence[2:-1])
            
            # Calculate maximum_sentence_length
            if length >= max_length:
                max_length = length
            
            # Append encoded sentence to the minibatch of X
            minibatch_x.append(one_hot)


        # data is a np.array of shape ('b', 's', 'w', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get 
        # ('b', 'SENTENCE_MAX_LENGTH', 'WORD_MAX_LENGTH', 'e')
        def numpy_fillna(data):
            """ This is a very useful function that fill the holes in our tensor """
            
            # Get lengths of each row of data
            lens = np.array([len(i) for i in data])

            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            out = np.zeros(shape=(mask.shape + (max_word_length, ALPHABET_SIZE)),
                           dtype='float32')

            out[mask] = np.concatenate(data)
            return out

        # Padding...
        minibatch_x = numpy_fillna(minibatch_x)

        return minibatch_x, np.array(minibatch_y)

    def load_to_ram(self, batch_size):
        """ Load n Rows from File f to Ram """
        # Returns True if there are still lines in the buffer, 
        # otherwise returns false - the epoch is over
        
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
        """ Returns Next Batch """
        
        # I realize this could be more 
        if dataset == TRAIN_SET:
            n_samples = 1600000 * 0.95
        elif dataset == VALID_SET:
            n_samples = 1600000 * 0.05
        elif dataset == TEST_SET:
            n_samples = 498
        
        # Number of batches / number of iterations per epoch
        n_batch = int(n_samples // batch_size)
        
        # Creates a minibatch, loads it to RAM and feed it to the network
        # until the buffer is empty
        for i in range(n_batch):
            if self.load_to_ram(batch_size):
                inputs, targets = self.make_minibatch(self.data)
                yield inputs, targets