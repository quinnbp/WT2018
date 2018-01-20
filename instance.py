class Instance:
    """ Instance class for WT2018 project. Contains non-binary label and ordered
        list of words for each tweet. """

    def __init__(self):
        self.label = None  # should be set to 0 or 1 when categorized
        self.fulltweet = ""
        self.wordlist = []

    def getFullTweet(self):
        return self.fulltweet

    def getWordList(self):
        return self.wordlist

    def getLabel(self):
        return self.label
