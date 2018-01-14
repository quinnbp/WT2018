import json
import pdb
import codecs
import pdb
import csv

def get_data():
    tweets = []
    files = ['Racist_Tweets_June2016_Dataset.csv', 'None_Tweets_June2016_Dataset.csv', 'Sexist_Tweets_June2016_Dataset.csv']

    for file in files:
        with open('./tweet_data/' + file) as f:
            read_csv = csv.reader(f, delimiter=',')
            for row in read_csv:
                tweets.append({
                    'id': row[0],
                    'text': row[1],
                    'label': row[2]
                })

    #pdb.set_trace()
    return tweets


if __name__=="__main__":
    tweets = get_data()
    males, females = {}, {}
    with open('./tweet_data/males.txt') as f:
        males = set([w.strip() for w in f.readlines()])
    with open('./tweet_data/females.txt') as f:
        females = set([w.strip() for w in f.readlines()])

    males_c, females_c, not_found = 0, 0, 0
    for t in tweets:
        if t['name'] in males:
            males_c += 1
        elif t['name'] in females:
            females_c += 1
        else:
            not_found += 1
    print(males_c, females_c, not_found)
    pdb.set_trace()
