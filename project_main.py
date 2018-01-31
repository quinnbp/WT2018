import random
import re

from nltk.corpus import stopwords
from bayes import BayesModel
from proximity import ProximityModel
from VotingClassifier.VotingClassifierObject import VotingModel

def main(tperc, seed, fpaths):
    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set = splitSets(tperc, seed, instances)

    b = BayesModel()
    p = ProximityModel()
    v = VotingModel()

    b.train(train_set)
    p.train(train_set)

    bayesResults = b.batchTest(test_set)
    proximityResults = p.batchTest(test_set)
    votingResults = v.predict(test_set)
    # TODO: call Sage's model

    confusionMatrices = [b.getConfusionMatrix(), p.getConfusionMatrix()]  # TODO add Sage's and Daniel's CMs
    weightResults(confusionMatrices, bayesResults, proximityResults, votingResults)

    return bayesResults, proximityResults, votingResults


def weightResults(confusionMatrices, bayesResults, proximityResults, votingResults):
    pass  # TODO


def splitSets(tperc, seed, instances):
    random.seed(seed)
    random.shuffle(instances)
    split = int(tperc * len(instances))
    return instances[:split], instances[split:]


def parseFiles(files):
    instances = []
    for f in files:
        instances += parseSingle(f)
    return instances


def parseSingle(f):
    instances = []

    stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff"]
    stopwords.extend(other_exclusions)
    stopwords = set(stopwords)  # set has faster existence test

    # this code written for files formatted like: labeled_data.csv

    line = file.readline()  # strip first line
    line = file.readline()
    while line != "":
        line = line.strip('"').rstrip('\n')
        split = line.split(",")

        # strip non-words out of tweet, split
        justWords = re.sub(r'[^a-zA-Z]+', '', split[6]).split()

        # strip stopwords
        for word in justWords:
            if word in stopwords:
                justWords.remove(word)

        # rejoining for fulltweet
        ft = ""
        for word in justWords:
            ft += word + " "

        # build instance
        i = Instance()
        i.label = split[5]
        i.wordlist = justWords
        i.fulltweet = ft

        print("parseSingle instance check: " + str(instance))
        instances.append(i)
        line = f.readline()

    return instances


def openFiles(filepaths):  # takes in file paths and attempts to open, fails if zero valid files
    files = []
    for fp in filepaths:
        try:
            f = open(fp, 'r')
            files.append(f)
        except FileNotFoundError:
            print("Readin: NonFatal: file " + str(fpath) + " not found.")

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
