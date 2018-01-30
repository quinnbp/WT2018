class Instance:
    """ Instance class for WT2018 project. Contains non-binary label and ordered
        list of words for each tweet. """

    def __init__(self):
        self.label = None
        self.fulltweet = ""
        self.wordlist = []

    def __str__(self):
        return str([self.label, self.fulltweet])

    def getFullTweet(self):
        return self.fulltweet

    def getWordList(self):
        return self.wordlist

    def getLabel(self):
        return self.label
