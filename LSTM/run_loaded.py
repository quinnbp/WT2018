#   TO RUN LOADED LSTM
def runLSTM(sentences):
    #sentences= []
    #for i in range(1000):
    #    sentences.append("this is a fucking offensive sentence goddamnit")
    print('Creating LSTM from load')
    if sentences is not None:
        for value in sentences:
            print('processing sentence: %s' % value)


    from lib_model.char_lstm import *

    network = LSTM()
    network.build()
    preds =[]
    if sentences is not None:
        for sentence in sentences:
            feeder = [sentence]
            preds.append(network.predict_sentences(feeder))
    return preds

