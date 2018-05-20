#   TO RUN LOADED LSTM
from lib_model.char_lstm import *

def runLSTM(sentences):

    print('Creating LSTM from load')
    # if sentences is not None:
    #     for value in sentences:
    #         print('processing sentence: %s' % value)

    network = LSTM()
    network.build()

    #Get sentences from tweet instances
    tweets = [t.getFullTweet for t in sentences]
    #tweets = [t.getCleanTweet for t in sentences]

    if sentences is not None:
        return network.predict_sentences(tweets)
