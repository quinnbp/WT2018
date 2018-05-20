#   TO RUN LOADED LSTM
def runLSTM(sentences):
    print('Creating LSTM from load')
    if sentences is not None:
        for value in sentences:
            print('processing sentence: %s' % value)

    
    from lib_model.char_lstm import *

    network = LSTM()
    network.build()

    if sentences is not None:
        return network.predict_sentences(sentences)
