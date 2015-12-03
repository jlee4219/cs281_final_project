import features
import read
import model
import selection
import preprocessing

import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def main():
  # Read the data from the text files
  begin = time.time()
  # No longer need to get vocab here
  vocab, train, test = read.read_tweets("../training_set_tweets.txt", "../test_set_tweets.txt")
  print "Train size:", len(train), "Test size:", len(test)
  print "Read data:", time.time() - begin

  # Preprocess tweets
  vocab, train, test = preprocessing.preprocess(train, test)

  # Assign ids to words
  vocab_list = list(vocab)
  vocab_list.sort()
  begin = time.time()
  vocab_dict = {}
  for i in range(len(vocab_list)):
      vocab_dict[vocab_list[i]] = i
  print "Assigned ids to words:", time.time() - begin


  # Build train and test set
  num_full_feats = len(vocab_list) + 10
  num_train_tweets = 0
  num_test_tweets = 0
  # num_train_tweets = np.count_nonzero(~np.isnan(train))
  # num_test_tweets = np.count_nonzero(~np.isnan(test))
  for author_id in train:
      num_train_tweets += len(train[author_id])
  for author_id in test:
      num_test_tweets += len(test[author_id])
  X_train = np.zeros((num_train_tweets, num_full_feats))
  y_train = np.zeros(num_train_tweets)
  X_test = np.zeros((num_test_tweets, num_full_feats))
  y_test = np.zeros(num_test_tweets)

  count = 0

  for author_id in train:
      for tweet in train[author_id]:
          X_train[count, :] = features.get_full_feats(tweet, vocab_dict)
          y_train[count] = author_id
          count += 1
  print count

  count = 0
  for author_id in test:
      for tweet in test[author_id]:
          X_test[count, :] = features.get_full_feats(tweet, vocab_dict)
          y_test[count] = author_id
          count += 1
  print count


  begin = time.time()
  clf = model.train(X_train, y_train)
  acc, my_acc, preds, scores = model.test(clf, X_test, y_test)
  print 'time:', time.time()-begin, 'acc:', acc, 'my_acc:', my_acc
  print 'preds:', preds
  print 'scores:', scores


main()



