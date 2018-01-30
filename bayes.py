import math

class BayesModel:
    def __init__(self):
        self.labels_dict = {}  # k = label, v = dict
        self.totals_dict = {}  # k = label, v = int

    def train(self, train_list):  # @param list of training instances, stop words removed
        train_dict = dict()
        labels = set()

        for inst in train_list:  # gathering and defaulting all labels
            l = inst.getLabel()
            if l not in labels:
                labels.add(l)
                train_dict[l] = []
            train_dict[l].append(inst)

        for label in train_dict.keys():  # establish wordcounts for each label
            wordcounts = {}
            totalwords = 0
            for instance in train_dict[label]:
                tweet = instance.getWordList()
                for word in tweet:
                    totalwords += 1
                    if word in wordcounts.keys():
                        wordcounts[word] += 1
                    else:
                        wordcounts[word] = 1

            self.labels_dict[label] = wordcounts
            self.totals_dict[label] = totalwords


    def testSingle(self, instance):
        words = instance.getWordList()

        prob = dict()
        total_labels = len(self.labels_dict.keys())
        for label in self.labels_dict.keys():
            prob[label] = 1.0 / total_labels

        for label in self.labels_dict.keys():
            lwc = self.labels_dict[label]
            for word in words:  # this is where the actual Bayesian probability happens
                if word in lwc.keys():
                    prob[label] += math.log((float(lwc[word] + 1)) / self.totals_dict[label])
                else:
                    prob[label] += math.log(1.0 / self.totals_dict[label])  # psuedocounts (nonzero)

        maxVal = 0
        currentLabel = ""
        for label in prob.keys():
            if prob[label] > maxVal:
                currentLabel = label
                maxVal = prob[label]

        return currentLabel


    def batchTest(self, instances):
        orderedGuesses = []
        for i in instances:
            orderedGuesses.append(testSingle(i))
        return orderedGuesses
