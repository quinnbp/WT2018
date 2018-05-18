#!/usr/bin/env python

""" bayes.py: Model class file for a Bayesian learning text-instance process of elimination module
        for WT2018 project. """

__author__ = "Quinn Barker-Plummer"
__email__  = "qbarkerp@oberlin.edu"
__status__ = "Development"

# Current accuracy: 9.9 %

import math


class BayesEliminationModel:
    def __init__(self):
        self.labels_dict = dict()  # k = label, v = dict
        self.totals_dict = dict()  # k = label, v = int
        self.cm = dict()

    def train(self, train_list):  # @param list of training instances
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


    def buildConfusionMatrix(self, test_list):
        guesses = self.batchTest(test_list)
        #print(guesses)
        actuals = []
        for inst in test_list:
            actuals.append(inst.getLabel())

        for idx in range(0, len(guesses)):
            cm_tuple = (guesses[idx], actuals[idx])
            if cm_tuple not in self.cm:
                self.cm[cm_tuple] = 0
            self.cm[cm_tuple] +=1


    def getConfusionMatrix(self, test_list):
        self.buildConfusionMatrix(test_list)
        return self.cm

    def testSingleElim(self, instance, remainingLabels):
        words = instance.getWordList()

        prob = dict()
        total_labels = len(remainingLabels)
        for label in remainingLabels:
            prob[label] = 1.0 / total_labels

        for label in remainingLabels:
            lwc = self.labels_dict[label]
            for word in words:  # this is where the actual Bayesian probability happens
                if word in lwc.keys():
                    prob[label] += - math.log((float(lwc[word] + 1)) / self.totals_dict[label])
                else:
                    prob[label] += - math.log(1.0 / self.totals_dict[label])  # psuedocounts (nonzero)

        maxVal = 0
        currentLabel = ""
        for label in prob.keys():
            # print(prob[label])
            if prob[label] > maxVal:
                currentLabel = label
                maxVal = prob[label]

        return currentLabel


    def testSingle(self, inst):
        rLabels = list(self.labels_dict.keys())
        while len(rLabels) > 1:
            badGuess = self.testSingleElim(inst, rLabels)
            #print(str(badGuess))
            rLabels.remove(badGuess)
        return rLabels[0]


    def batchTest(self, instances):
        orderedGuesses = []
        for i in instances:
            orderedGuesses.append(self.testSingle(i))
        return orderedGuesses
