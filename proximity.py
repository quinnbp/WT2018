from readin import prep as run_readin
import math

class ProximityModel:
    def __init__(self):
        self.train_dict = dict()  # k = frozenset (keywords), v = int (score)
        self.label_dict = dict()  # k = label, v = corresponding int
        self.trained = False


    def main(self, fpaths, tperc, seed):
        stopWords = set()
        try:
            stop_file = open("stopWords.txt", "r")
            for line in stop_file:
                stopWords.add(line.lower().rstrip("\n"))
        except IOError:
            print("Proximity: NonFatal: stopWords file not found")

        train_list, test_list, labels = run_readin(fpaths, tperc, seed, stopWords)
        self.build_label_dict(labels)
        self.train(train_list)


    def build_label_dict(self, labels):
        # assigns int classification score for each label
        for idx in range(0, len(labels)):
            self.label_dict[labels[idx]] = idx


    def train(self, train_list):
        # stopWords removed in readin
        for inst in train_list:
            key = frozenset(inst.wordlist)
            self.train_dict[key] = self.label_dict[inst.label]


    def test(self, test_list):
        guess_list = []
        for inst in test_list:
            guess_list.append(self.test_single(inst))


    def test_single(self, inst):
        guess_int = 0
        denom = 1
        for word in inst.wordlist:
            for trkey in self.train_dict.keys():
                if word in trkey:
                    guess_int += self.train_dict[trkey]
                    denom += 1

        return math.round(float(guess_int) / denom)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        # not enough args to make tperc or seed
        raise IndexError("Readin: Fatal: Not enough given arguments.")
    else:
        tperc = float(sys.argv[1])
        seed = int(sys.argv[2])
        fpaths = sys.argv[3:]

        p = ProximityModel()
        p.main(fpaths, tperc, seed)