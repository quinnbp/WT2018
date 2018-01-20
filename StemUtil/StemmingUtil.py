import sys
sys.path.append("./nltk")

import nltk
from nltk.stem.snowball import SnowballStemmer

def parseTokens(string):
	return nltk.word_tokenize(string) 

def createStems(tokens):
	stemmer = SnowballStemmer("english")

	stems = []
	for token in tokens:
		stems.append(stemmer.stem(token))

	return stems
