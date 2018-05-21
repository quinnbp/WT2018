#   TO TEST RUNNING LOADED LSTM
from LSTM.lib_model.char_lstm import *

def main():
    sentences = ["jeez", "ell"]
    print('Creating LSTM from load')
    if sentences is not None:
        for value in sentences:
            print('processing sentence: %s' % value)

    network = LSTM()
    network.build()

    if sentences is not None:
        network.predict_sentences(sentences)


if __name__ == '__main__':
    main()