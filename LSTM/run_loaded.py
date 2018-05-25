from LSTM.lib_model.char_lstm import *

#   TO RUN LOADED LSTM
def runLSTM(sentences):
    #sentences= []
    #for i in range(1000):
    #    sentences.append("this is a fucking offensive sentence goddamnit")
    print('Creating LSTM from load')
    # if sentences is not None:
    #     for value in sentences:
    #         print('processing sentence: %s' % value)

    network = LSTM()
    network.build()
    preds =[]
    feeder = []
    i = 0
    sentences = [t.getFullTweet() for t in sentences]

    if sentences is not None:
        for sentence in sentences:
            feeder.append(sentence)
            if i % 1000 == 0:
                preds.append(network.predict_sentences(feeder))
                feeder = []
                print(preds)
            i += 1

    return preds

