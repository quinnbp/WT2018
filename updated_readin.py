import random
from nltk.corpus import stopwords
from bayes import BayesModel
from proximity import ProximityModel

def main(tperc, seed, fpaths):
    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set = splitSets(tperc, seed, instances)

    b = BayesModel()
    p = ProximityModel()

    b.train(train_set)
    p.train(train_set)

    bayesResults = b.batchTest(test_set)
    proximityResults = p.batchTest(test_set)

    return bayesResults, proximityResults


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
    stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    pass  # TODO: some processing, depends on input set


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