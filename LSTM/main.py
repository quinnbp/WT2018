#   TO TEST RUNNING LOADED LSTM
if __name__ == '__main__':
    sentences = ["jeez","ell"]
    print('Creating LSTM from load')
    if sentences is not None:
        for value in sentences:
            print('processing sentence: %s' % value)

    
    from lib_model.char_lstm import *

    network = LSTM()
    network.build()

    if sentences is not None:
       network.predict_sentences(sentences)
