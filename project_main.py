import random
import re

from nltk.corpus import stopwords
from bayes import BayesModel
from proximity import ProximityModel
from lstm import LSTM
from VotingClassifier.VotingClassifierObject import VotingModel

def main(tperc, seed, fpaths):
    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set1, test_set2 = splitSets(tperc, seed, instances)

    # initialize all models
    b = BayesModel()
    p = ProximityModel()
    v = VotingModel()
    r = LSTM() 

    # train all models (except voting, loaded)
    b.train(train_set)
    p.train(train_set)
    r.train(train_set) 

    # run models and store first set of results
    b.batchTest(test_set1)
    p.batchTest(test_set1)
    v.predict(test_set1)
    r.predict(test_set1) 

    # get confusion matrices for first set of results
    confusionMatrices = [b.getConfusionMatrix(), p.getConfusionMatrix(), v.getConfusionMatrix(), r.getConfusionMatrix()]

    # weight second set of results, using first
    weightingInput = [
        [confusionMatrices[0] ,b.batchTest(test_set2)],
        [confusionMatrices[1], p.batchTest(test_set2)],
        [confusionMatrices[2], v.predict(test_set2)],
        [confusionMatrices[3], r.predict(test_set2)] 
    ]

    guesses = weightResults(weightingInput)

    return guesses


def weightResults(weightingInput):
    # this code written for files formatted like: labeled_data.csv
    weightingList = []
    for pair in weightingInput:
        cm = pair[0]
        output = pair[1]

        accuracyFor = dict()
        for pred in [0, 1, 2]:  # should un-hard-code this at some point
            total = 0
            correct = 0
            for act in [0, 1, 2]:
                current = cm[(pred, act)]
                if current is None:
                    current = 0
                if pred == act:
                    correct += current
                total += current

            accuracyFor[pred] = float(correct) / total

        weightingList.append(accuracyFor, output)

    return predictAll(weightingList)

def predictAll(weightingList):
    guesses = []
    for idx in range(0, len(weightingList[0][1])):  # for each instance
        votes = dict()
        for mpair in weightingList:  # for each model's results
            voteFor = mpair[1][idx]  # what the model guessed
            votes[voteFor] += mpair[0][votefor]  # how 'much' of a vote that should be (by accuracy)

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
        i.label = int(split[5])
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
