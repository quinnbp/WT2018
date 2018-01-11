import requests
import csv
import openpyxl

#Gets a list of tweet ids, and returns a list of those tweets
def get_tweets(tweetid_list):
    #tweetid_list = ["551769698917167104", "551787027725705216", "551799120080293888"]

    all_tweets = []
    i = 0
    contains = False

    while (i < len(tweetid_list)):

        #Get tweet page and store all elements as a list of words
        response = requests.get("http://www.twitter.com/statuses/" + tweetid_list[i])
        words = response.text.split()

        #print ("Tweet ID is " + tweetid_list[i])

        #Tweet in list format
        l_tweet = []
        w_index = 0

        #Get tweet out of webpage in list format
        while (w_index < len(words)):
            word = words[w_index]

            if "<title>" in word:
                contains = True

            if contains:
                l_tweet.append(word)

            if "</title>" in word:
                contains = False
                break

            w_index += 1

        #print("Current tweet in list version is " + str(l_tweet))

        #Filter beginning of tweet -- Removes "<title> User on Twitter:"
        for j in range(len(l_tweet)):
            if "&quot;" in l_tweet[0]:
                break
            l_tweet.remove(l_tweet[0])

        #Turn tweet into a string, format it and add it to list of tweets in string format -- Removes "&quot;" and "</title>"
        new_string = ' '.join(l_tweet)

        new_string2 = new_string.replace("&quot;", "")
        tweet_string = new_string2.replace("</title>", "")

        if i % 100 == 0:
            print("Fetching " + str(i) + " tweets\n")
            print("Current tweet is: " + tweet_string)
            #print()

        all_tweets.append(tweet_string)

        i += 1

    #print(all_tweets)
    return all_tweets

if __name__ == '__main__':

    tweetid_list = []
    tweet_labels = []
    all_tweets = []

    #EXCEL READING --------------

    wb = openpyxl.load_workbook("None_Tweets_June2016.xlsx")
    sheet = wb.get_sheet_by_name("Sheet1")

    #Stores lists of Tweet IDs and Tweet Labels
    for cell in sheet.rows:
        tweetid_list.append(cell[0].value)
        tweet_labels.append(cell[1].value)

    #------------------------------

    #Get the tweets givem the list of tweet IDs
    all_tweets = get_tweets(tweetid_list)

    #print(tweetid_list)

    #Join together all three lists
    tweet_file = []
    for i in range (len(tweetid_list)):
        tweet_file.append([tweetid_list[i], all_tweets[i], tweet_labels[i]])

    #Write the new tweets in a file
    with open("None_Tweets_June2016_Dataset.csv", "w") as newfile:
        write_csv = csv.writer(newfile)
        write_csv.writerows(tweet_file)

    #print(tweet_file)