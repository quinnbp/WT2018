import math


class BayesModel:
    def __init__(self):
        self.labels_dict = {}  # k = label, v = dict
        self.totals_dict = {}  # k = label, v = int

        self.cm = {}

    def train(self, train_list):  # @param list of training instances
        stopWords = set()
        try:
            stop_file = open("stopWords.txt", "r")
            for line in stop_file:
                stopWords.add(line.lower().rstrip("\n"))
        except IOError:
            print("Bayes: NonFatal: stopWords file not found")

        train_dict = dict()
        labels = set()

        for inst in train_list:  # gathering and defaulting all labels
            l = inst.getLabel()
            if l not in labels:
                labels.add(l)
                train_dict[l] = [inst]
            else:
                train_dict[l].append(inst)

        for label in train_dict.keys():  # establish wordcounts for each label
            wordcounts = {}
            totalwords = 0
            for instance in train_dict[label]:
                tweet = instance.getWordList()
                for word in tweet:
                    if word not in stopWords:  #
                        totalwords += 1
                        if word in wordcounts.keys():
                            wordcounts[word] += 1
                        else:
                            wordcounts[word] = 1

            self.labels_dict[label] = wordcounts
            self.totals_dict[label] = totalwords

    def test(self, instance):
        true_label = instance.getLabel()
        words = instance.getWordList()

        prob = dict()  # should only be two labels
        prob[0] = 0.5
        prob[1] = 0.5

        for label in self.labels_dict.keys():
            lwc = self.labels_dict[label]
            for word in words:  # this is where the actual Bayesian probability happens
                if word in lwc.keys():
                    prob[label] += math.log((float(lwc[word] + 1)) / self.totals_dict[label])
                else:
                    prob[label] += math.log(float(1) / self.totals_dict[label])  # psuedocounts (nonzero)

        # predicted, actual
        if prob[0] >= prob[1]:
            cm_tuple = (0, true_label)
        else:  # prob[1] > prob[0]
            cm_tuple = (1, true_label)

        if cm_tuple in self.cm.keys():  # add to ongoing confusion matrix
            self.cm[cm_tuple] += 1
        else:
            self.cm[cm_tuple] = 1

    def guess(self, instance):
        words = instance.getWordList()

        prob = dict()  # should only be two labels
        prob[0] = 0.5
        prob[1] = 0.5

        for label in self.labels_dict.keys():
            lwc = self.labels_dict[label]
            for word in words:  # this is where the actual Bayesian probability happens
                if word in lwc.keys():
                    prob[label] += math.log((float(lwc[word] + 1)) / self.totals_dict[label])
                else:
                    prob[label] += math.log(float(1) / self.totals_dict[label])  # psuedocounts (nonzero)

        if prob[0] >= prob[1]:  # return more likely label
            return 0
        else:  # prob[1] > prob[0]
            return 1

    def assess(self):
        total_runs = 0
        correct = 0
        for tup in self.cm.keys():
            total_runs += self.cm[tup]
            if tup[0] == tup[1]:
                correct += self.cm[tup]

        return float(correct) / total_runs

    def batchTest(self, instances):
        for i in instances:
            self.test(i)
        print(self.assess())
