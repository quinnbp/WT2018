import random
import sys
from instance import Instance

def readin(files, tperc, seed):
    # Takes a list of file objects and returns train/test lists of instances
    total = []
    labels = []
    for f in files:
        file_total, file_labels = parsefile(f)

        total.extend(file_total)  # add new instances to overall total
        labels.extend(file_labels)  # add new labels to overall labels

    tcut = int(tperc * len(total))
    shuffled_total = shuffler(total, seed)

    train = shuffled_total[:tcut]  # cut based on train percentage
    test = shuffled_total[tcut:]

    return train, test, labels  # should be mixed-label sets

def parsefile(f):
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
        n.wordlist = lsplit[1].split()  # cut between words

        total.append(n)
        line = f.readline()

    return total, labels


def shuffler(theset, seed):
    # randomizes a list according to a given seed
    random.seed(seed)
    random.shuffle(theset)
    return theset

if __name__ == "__main__":
    if len(sys.argv) < 4:
        # not enough args to make tperc or seed
        raise IndexError("Readin: Fatal: Not enough given arguments.")
    else:
        tperc = float(sys.argv[1])
        seed = int(sys.argv[2])
        fpaths = sys.argv[3:]

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
            readin(files, tperc, seed)





