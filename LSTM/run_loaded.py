from lib_model.char_lstm import *
from sklearn.model_selection import train_test_split
import pandas as pd
import re

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
            if i % 50 == 0 or i % len(sentences) == 0:
                network.predict_sentences(feeder)
                preds.append(network.predlist)
                feeder = []
            i += 1
    return network.predlist

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def process_data(fname):

    f = open(fname, 'r')

    # Read inputs using pandas
    df = pd.read_csv(f)

    raw_tweets = df.tweet.tolist()
    labels = df['class'].astype(int)

    raw_tweets = [preprocess(t) for t in raw_tweets]

    # UNCOMMENT IF ATTEMPTING TO PREDICT TWEETS OF ONLY ONE LABEL
    # tweets = []
    #
    # for tweet, label in zip(raw_tweets, labels):
    #     if label == 0:                              #0: Hate Speech, 1: Offensive and 2: Neutral
    #         tweets.append(tweet)

    X, y = train_test_split(raw_tweets, test_size=0.2)

    print(y)
    print("Size of training set is", len(y))
    return y

if __name__ == "__main__":
    name = "./LSTM/Datasets/labeled_data.csv"
    test = process_data(name)
    runLSTM(test)