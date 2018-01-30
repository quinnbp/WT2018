from readin import prep as run_readin
import math

class ProximityModel:
    def __init__(self):
        self.wordmap = dict()


    def train(self, train_list):  # instances must be int-labeled
        for inst in train_list:
            for word in inst.getWordList():
                if word not in prep_dict:
                    self.wordmap[word] = [inst.getLabel(), 1]
                else:
                    self.wordmap[word][0] += inst.getLabel()
                    self.wordmap[word][1] += 1
        for k in self.wordmap:
            self.wordmap[k] = float(self.wordmap[k][0]) / self.wordmap[k][1]


    def batchTest(self, test_list):
        orderedGuesses = []
        for inst in test_list:
            guess_list.append(self.test(inst))
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
        return math.round(float(guess) / denom)

