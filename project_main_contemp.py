import nltk
from nltk.stem.porter import *
import re
import pandas as pd
import random
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from instance import Instance
from weighting import voting
#from bayes import BayesModel
from proximity import ProximityModel
#from lstm import LSTM
from VotingClassifier.VotingClassifierObject import VotingModel
from confusion_matrix import ConfusionMatrix


def main(tperc, seed, fpaths):
    """Parses files, trains the models, tests the models,
    creates the weights, makes predictions and evaluates results"""

    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set1, test_set2 = splitSets(tperc, seed, instances)

    # Initialize all models

    p = ProximityModel()
    v = VotingModel()
    # b = BayesModel()
    # r = LSTM()

    # Train all models

    p.train(train_set)
    v.train(train_set)
    # b.train(train_set)
    # r.train(train_set)

    # Run models and store first set of results

    p_pred = p.batchTest(test_set1)
    v_pred = v.batchTest(test_set1)
    # b_pred = b.batchTest(test_set1)
    # r_pred = r.batchTest(test_set1)

    # Get confusion matrices for first set of results

    test_set1_labels = [i.getLabel() for i in test_set1]
    p_cm = ConfusionMatrix(test_set1_labels, p_pred, "Proximity")
    v_cm = ConfusionMatrix(test_set1_labels, v_pred, "Voting")
    # b_cm = ConfusionMatrix(test_set1_labels, b_pred, "Bayes")
    # r_cm = ConfusionMatrix(test_set1_labels, r_pred, "LSTM")

    confusionMatrices = [p_cm, v_cm]
    # confusionMatices = [p_cm, v_cm, b_cm, r_cm]

    # Weight second set of results, using first set
    weightingInput = [
        [confusionMatrices[0], p.batchTest(test_set2)],
        [confusionMatrices[1], v.batchTest(test_set2)],
        # [confusionMatrices[2] ,b.batchTest(test_set2)],
        # [confusionMatrices[3], r.batchTest(test_set2)],
    ]

    # Get the weighting results
    guesses = voting(weightingInput)
    # print(guesses)

    # Compare results with actual labels
    test_set2_labels = [i.getLabel() for i in test_set2]
    evaluate_accuracy(guesses, test_set2_labels)

    # Store second set of tweets and guesses
    test_set2_tweets = [t.getFullTweet() for t in test_set2]
    store_new_labels(test_set2_tweets, guesses, test_set2_labels)


def store_new_labels(t2tweets, guesses, labels):
    """Creates a csv document that stores the tweets tested and their predicted labels"""

    with open("FinalModel_Predictions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet", "class", "label"])
        for tweet, guess, label in zip(t2tweets, guesses, labels):
            writer.writerow([tweet, guess, label])

def evaluate_accuracy(guesses, labels):
    """Compares labels returned by weighting method to labels in dataset,
        produces a confusion matrix and prints the recall/precision/f1 score table
    """
    print(classification_report(labels, guesses))

    # Confusion matrix making and storing
    cfm = confusion_matrix(labels, guesses)

    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = cfm[i, :] / float(cfm[i, :].sum())

    names = ['Hate', 'Offensive', 'Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True,
                    fmt='.2f')
    plt.ylabel(r'True categories', fontsize=14)
    plt.xlabel(r'Predicted categories', fontsize=14)
    plt.tick_params(labelsize=12)

    #print(cfm)
    #print(matrix_proportions)
    plt.savefig("FinalModel_ConfusionMatrix.pdf")
    print("Stored confusion matrix!")


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
    raw_tweets = df.tweet
    labels = df['class'].astype(int)
    instances = []

    # Process tweets and create instances
    for tweet, label in zip(raw_tweets, labels):

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
