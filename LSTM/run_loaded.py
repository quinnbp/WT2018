#   TO RUN LOADED LSTM
def runLSTM(sentences):
    #sentences= []
    #for i in range(1000):
    #    sentences.append("this is a fucking offensive sentence goddamnit")
    print('Creating LSTM from load')

    from lib_model.char_lstm import *

    network = LSTM()
    network.build()
    preds =[]
    feeder = []
    i =0
    if sentences is not None:
        for sentence in sentences:
            feeder.append(sentence)
            if i % 100 == 0 or i % len(sentences) == 0:
                network.predict_sentences(feeder)
                preds.append(network.predlist)
                feeder = []
            i+=1;
    return network.predlist

#TEST
#if __name__ == '__main__':
    #sentence = "test"
    #l = []
    #for i in range(100):
    #    l.append(sentence)
    #runLSTM(l)

