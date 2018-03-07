import nltk
from nltk.stem.porter import *
import re
import pandas as pd
import random
import sys

from instance import Instance
# from bayes import BayesModel
# from proximity import ProximityModel
# from lstm import LSTM
from VotingClassifier.VotingClassifierObject import VotingModel


def main(tperc, seed, fpaths):
    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set1, test_set2 = splitSets(tperc, seed, instances)

    # initialize all models

    # b = BayesModel()
    # p = ProximityModel()
    v = VotingModel()
    # r = LSTM()

    # train all models (except voting, loaded)

    # b.train(train_set)
    # p.train(train_set)
    v.train(train_set)
    # r.train(train_set)

    # run models and store first set of results

    v.predict(test_set1)
    # r.predict(test_set1)

    # get confusion matrices for first set of results

    # confusionMatrices = [b.getConfusionMatrix(), p.getConfusionMatrix(), v.getConfusionMatrix(), r.getConfusionMatrix()]

    # patch code
    # confusionMatrices = [b.getConfusionMatrix(test_set1), p.getConfusionMatrix(test_set1)]
    confusionMatrices = [v.getConfusionMatrix(test_set1)]

    # weight second set of results, using first
    weightingInput = [
        # [confusionMatrices[0] ,b.batchTest(test_set2)],
        # [confusionMatrices[1], p.batchTest(test_set2)],
        [confusionMatrices[0], v.predict(test_set2)],
        # [confusionMatrices[3], r.predict(test_set2)]  # patch comment
    ]

    # TODO Write weightResults
    guesses = weightResults(weightingInput)
    print(guesses)

    return guesses


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


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    stemmer = PorterStemmer()
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    # tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def main_parser(f):
    """"
        @input file
        @output list of instance objects

        Reads files in the format as labeled_data.csv as a pandas dataframe
        This means that it contains a top row with the words tweets | class,
        so they can be referenced easily.

        Creates instance objects with the full text, the tokenized text and the label
    """

    # Read inputs using pandas
    df = pd.read_csv(f)
    raw_tweets = df.tweets
    labels = df['class'].astype(int)
    instances = []

    # Process tweets and create instances
    for tweet, label in (raw_tweets, labels):

        # Raw tweet and label
        i = Instance()
        i.label = label
        i.fulltweet = tweet

        # Get just text
        clean_tweet = preprocess(tweet)
        i.clean_tweet = clean_tweet

        # Tokenize tweet
        tokenized_tweet = basic_tokenize(clean_tweet)
        # stemmed_tweet = tokenize(clean_tweet)
        i.wordlist = tokenized_tweet

        instances.append(i)

    return instances


def predictAll(weightingList):
    guesses = []
    for idx in range(0, len(weightingList[0][1])):  # for each instance
        votes = dict()
        for mpair in weightingList:  # for each model's results
            voteFor = mpair[1][idx]  # what the model guessed
            if voteFor in votes.keys():
                votes[voteFor] += mpair[0][voteFor]  # how 'much' of a vote that should be (by accuracy)
            else:
                votes[voteFor] = mpair[0][voteFor]
        maxVal = 0
        vote = None
        for label in votes.keys():  # get max voted
            if votes[label] > maxVal:
                maxVal = votes[label]
                vote = label

        guesses.append(vote)  # guess for instance

    return guesses


def splitSets(tperc, seed, instances):
    random.seed(seed)
    random.shuffle(instances)
    split = int(tperc * len(instances))
    second_split = int(split + float(len(instances) - split) / 2)

    return instances[:split], instances[split:second_split], instances[second_split:]


def parseFiles(files):
    instances = []
    for f in files:
        instances += main_parser(f)
    return instances


def openFiles(filepaths):  # takes in file paths and attempts to open, fails if zero valid files
    files = []
    for fp in filepaths:
        try:
            f = open(fp, 'r')
            files.append(f)
        except FileNotFoundError:
            print("Readin: NonFatal: file " + str(fp) + " not found.")

    if len(files) == 0:
        raise FileNotFoundError("Readin: Fatal: No Files Found")
    else:
        return files


if __name__ == "__main__":
    if len(sys.argv) < 4:
        # not enough args to make tperc or seed
        raise IndexError("Readin: Fatal: Not enough given arguments.")
    else:
        tperc = float(sys.argv[1])
        seed = int(sys.argv[2])
        fpaths = sys.argv[3:]

        main(tperc, seed, fpaths)
