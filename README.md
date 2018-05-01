# Ensemble Learning for Tweet Classification of Hate Speech and Offensive Language - Winter/Spring Project 2018

## Idea

The aim of this project is to study how machine learning can be used to develop statistical models that can automatically identify whether the contents of a given Tweet contain offensive language, or even hate speech, based on the words used and the context they form.  Although an important goal of its own, the ultimate vision of this research is that such a model might be used in the future to enable intelligent agents (e.g., conversational chatbots) to discern between acceptable and unacceptable conversational patterns and avoid learning their own undesirable behaviors while interacting with people.

## Method

Ensemble learning through weighted voting systems, which include precision score, CEN score and equal voting. Councilor classifiers are:
- Proximity Model
- Voting Model
- LSTM Model
- Bayes Model

## Dataset

The project will use publicly available data in the form of collections of Tweets from Twitter (where all users agree that the contents of their messages shared in Tweets are publicly available once sent, https://twitter.com/en/tos), as well as coded categorizations of those Tweets (e.g., innocuous, offensive, hate speech).  Where possible, we will also furtherstrip any potentially identifying information (users, tags, etc.) from the data.  Overall, this data will be used both (1) as examples to train the machine learning models so that,when given a new Tweet, a computer program will subsequently be able to identify the category that most probably fits that the content of the Tweet, and (2) to evaluate the performance and predictions of the trained models in order to optimize performance on evaluating whether or not a Tweet contains offensive language or hate speech.

- Original dataset (for hate speech, offensive and neutral tweets): Datasets/labeled_data.csv
- Alternative dataset (for sexist, racist and neutral tweets): Datasets/All_Tweets_June2016_Dataset.csv

## Dependencies

- nltk
- numpy
- pandas
- sklearn
- scipy
- tensorflow
- gensim
- vaderSentiment
- matplotlib
- seaborn
	
## Instructions

1. Run project_main.py, commenting out undesired models 
	- Train models and make predictions
	- Store predictions for test_set_1 and test_set_2
	- Sample run:
	```
	python3 project_main.py 0.8 1234 CEN_Precision Datasets/labeled_data.csv
	```
2. Run tests.py 
	- Open predictions and create confusion matrices
	- Run multiple weighted voting methods, print classification report and store confusion matrices 
	

