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


    def train(self, train_list):  # instances must be int-labeled
        for inst in train_list:
            for word in inst.getWordList():
                if word not in self.wordmap:
                    self.wordmap[word] = [inst.getLabel(), 1]
                else:
                    self.wordmap[word][0] += inst.getLabel()
                    self.wordmap[word][1] += 1
        for k in self.wordmap:
            self.wordmap[k] = float(self.wordmap[k][0]) / self.wordmap[k][1]


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
        print(orderedGuesses)
        return orderedGuesses


    def test(self, inst):
        guess = 0
        denom = 0
        for word in inst.getWordList():
            if word in self.wordmap:  # if we've seen this word
                guess += self.wordmap[word]
                denom += 1
        if denom == 0:
            denom = 1
        return round(float(guess) / denom)

