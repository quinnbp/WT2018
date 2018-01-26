import random
import sys

from instance import Instance
from StemUtil.StemmingUtil import StemmingUtil as St

def prep(fpaths, tperc, seed, stopWords):
    files = []
    for path in fpaths:
        try:
            f = open(path)
            files.append(f)
        except FileNotFoundError:
            # missing one file
            print("Readin: Nonfatal: file " + str(path) + " not found.")

    if len(files) == 0:
        # found no files
        raise FileNotFoundError("Readin: Fatal: No train/test files found")
    else:
        # found at least one file
        return readin(files, tperc, seed, stopWords)


def readin(files, tperc, seed, stopWords):
    # Takes a list of file objects and returns train/test lists of instances
    total = []
    labels = []
    for f in files:
        file_total, file_labels = parsefile(f, stopWords)

        total.extend(file_total)  # add new instances to overall total
        labels.extend(file_labels)  # add new labels to overall labels

    tcut = int(tperc * len(total))
    shuffled_total = shuffler(total, seed)

    train = shuffled_total[:tcut]  # cut based on train percentage
    test = shuffled_total[tcut:]

    return train, test, labels  # should be mixed-label sets


def parsefile(f, stopWords):
    labels = []
    total = []

    line = f.readline()
    while line != "":
        line = line.rstrip('\n').replace('"', '')
        lsplit = line.split(",")
        n = Instance()

        n.label(lsplit[2])  # assign label and record
        if n.getLabel() not in labels:
            labels.append(n.getlabel())

        n.fulltweet = lsplit[1]  # cut between CSV cols

        words = St.parseTokens(n.fulltweet)
        wordlist = []
        for w in words:
            if w not in stopWords:
                wordlist.append(w)

        n.wordlist = StemmingUtil.createStems(wordlist)
        # in this line, the instance's wordlist is turned into stems
        # this means that various words lose plurals/ structural indicators
        # which helps to *not* distinguish between words that are functionally
        # the same

        total.append(n)
        line = f.readline()

    return total, labels


def shuffler(theset, seed):
    # randomizes a list according to a given seed
    random.seed(seed)
    random.shuffle(theset)
    return theset





