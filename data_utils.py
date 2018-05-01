import random, csv

def reshape_lines(lines):
    # Hacky function to make it easier to process the data
    data = []
    for l in lines:
        # 
        split = l.split('","')
        data.append((split[0][1:], split[-1][:-2]))
    return data

def save_csv(out_file, data):
    # Save a file
    with open(out_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('Data saved to file: %s' % out_file)

def shuffle_datasets(valid_perc=0.05):
    """ Shuffle the datasets """
    # TRAIN_SET and TEST_SET are respectively the path for 
    # training.1600000.processed.noemoticon.csv and
    # testdata.manual.2009.06.14.csv, this function will create two 
    # new files called "valid_set.csv" and "train_set.csv".
    
    # Make sure the paths exists, otherwise send some help messages...
    assert os.path.exists(TRAIN_SET), 'Download the training set at ' \
                                      'http://help.sentiment140.com/for-students/'
    assert os.path.exists(TEST_SET), 'Download the testing set at ' \
                                     'http://help.sentiment140.com/for-students/'

    # Create training and validation set - We take 5% of the training set 
    # for the validation set by default
    print('Creating training & validation set...')

    with open(TRAIN_SET, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines_train = lines[:int(len(lines) * (1 - valid_perc))]
        lines_valid = lines[int(len(lines) * (1 - valid_perc)):]

    save_csv(PATH + 'datasets/valid_set.csv', reshape_lines(lines_valid))
    save_csv(PATH + 'datasets/train_set.csv', reshape_lines(lines_train))

    print('Creating testing set...')

    with open(TEST_SET, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    save_csv(PATH + 'datasets/test_set.csv', reshape_lines(lines))
    print('All datasets have been created!')
    
# Once this is done, we rename the new training and testing set...
TRAIN_SET ='Datasets/All_Tweets_June2016_Dataset.csv'
TEST_SET ='Datasets/labeled_data.csv'
VALID_SET ='Datasets/labeled_data.csv'