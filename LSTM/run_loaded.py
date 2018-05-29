from lib_model.char_lstm import *
from sklearn.model_selection import train_test_split
import pandas as pd

#   TO RUN LOADED LSTM
def runLSTM(sentences):
    #sentences= []
    #for i in range(1000):
    #    sentences.append("this is a fucking offensive sentence goddamnit")
    print('Creating LSTM from load')

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
            i += 1
    return network.predlist


def process_data(fname):

    f = open(fname, 'r')

    # Read inputs using pandas
    df = pd.read_csv(f)

    raw_tweets = df.tweet.tolist()

    X, y = train_test_split(raw_tweets, test_size=0.2)

    return y

if __name__ == "__main__":
    name = "./LSTM/Datasets/labeled_data.csv"
    test = process_data(name)
    runLSTM(test)