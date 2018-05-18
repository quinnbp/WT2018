#!/usr/bin/env python

""" proximity.py: Model class file for a similarity/keyword based text-instance comparison
        module for WT2018 project. """

__author__ = "Quinn Barker-Plummer"
__email__  = "qbarkerp@oberlin.edu"
__status__ = "Development"

import math

class ProximityModel:
    def __init__(self):
        self.wordmap = dict()
        self.cm = dict()
        self.stopwords = set()

    def buildStopWords(self):
        try:
            f = open("stopWords.txt", 'r')
        except FileNotFoundError:
            print("BayesElim: NonFatal: No stopwords file found.")
            return  # if no file found

        line = f.readline()
        while line != "":
            self.stopwords.add(str(line.rstrip('\n')))
            line = f.readline()

    def train(self, train_list):
        self.buildStopWords()
        print(self.stopwords)

        labels = []  # collect labels
        for inst in train_list:
            if inst.getLabel() not in labels:
                labels.append(inst.getLabel())

        for inst in train_list:  # build map on labels
            for word in inst.getWordList():
                if word not in self.stopwords:
                    if word not in self.wordmap:  # if we need to add a word
                        self.wordmap[word] = dict()
                        for label in labels:
                            self.wordmap[word][label] = 0

                    self.wordmap[word][inst.getLabel()] += 1  # add label counter

        for word in self.wordmap:  # find most common label by word
            maxLabel = ""
            currentMax = 0
            for label in self.wordmap[word].keys():
                count = self.wordmap[word][label]
                if count > currentMax:
                    maxLabel = label
                    currentMax = count

            self.wordmap[word] = maxLabel  # assign to each word its max label

    def test(self, inst):
        encounteredLabels = dict()
        for word in inst.getWordList():
            if word not in self.stopwords:
                if word in self.wordmap:
                    if word not in encounteredLabels:
                        encounteredLabels[self.wordmap[word]] = 0
                    encounteredLabels[self.wordmap[word]] += 1

        maxLabel = ""
        currentMax = 0
        for label in encounteredLabels.keys():
            count = encounteredLabels[label]
            if count > currentMax:
                maxLabel = label
                currentMax = count

        return maxLabel

    def buildConfusionMatrix(self, test_list):
        guesses = self.batchTest(test_list)
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

    def batchTest(self, test_list):
        orderedGuesses = []
        for inst in test_list:
            orderedGuesses.append(self.test(inst))
        #print(orderedGuesses)
        return orderedGuesses


