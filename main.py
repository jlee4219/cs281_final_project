import features
import read
import model
import selection
import preprocessing
import feature_selection

import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def main():
  # Read the data from the text files
  begin = time.time()
  vocab, train_raw, test_raw = read.read_tweets("../training_set_tweets.txt", "../test_set_tweets.txt")
  print "Num of Train users:", len(train_raw), "Num of Test users:", len(test_raw)
  print "Read data:", time.time() - begin

  # Preprocess the data
  begin = time.time()
  vocab, train_word, test_word, train_char, test_char = preprocessing.preprocess(train_raw, test_raw)
  print "Preprocessed the data", time.time() - begin

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

  # Build train and test set
  num_full_feats = len(vocab_list) + 10
  num_train_tweets = 0
  num_test_tweets = 0
  # num_train_tweets = np.count_nonzero(~np.isnan(train))
  # num_test_tweets = np.count_nonzero(~np.isnan(test))
  for author_id in train_word:
      num_train_tweets += len(train_word[author_id])
  for author_id in test_word:
      num_test_tweets += len(test_word[author_id])
  X_train = np.zeros((num_train_tweets, num_full_feats))
  y_train = np.zeros(num_train_tweets)
  X_test = np.zeros((num_test_tweets, num_full_feats))
  y_test = np.zeros(num_test_tweets)

  count = 0

  for author_id in train_word:
      for tweet in train_word[author_id]:
          X_train[count, :] = features.get_full_feats(tweet, vocab_dict)
          y_train[count] = author_id
          count += 1
  print count

  count = 0
  for author_id in test_word:
      for tweet in test_word[author_id]:
          X_test[count, :] = features.get_full_feats(tweet, vocab_dict)
          y_test[count] = author_id
          count += 1
  print count

  begin = time.time()
  feats = feature_selection.select_features(X_train, y_train, np.zeros(num_full_feats), 100, "dia")
  X_train = X_train[:, feats]
  X_test = X_test[:, feats]
  print "Features selected:", time.time() - begin

  begin = time.time()
  clf = model.train(X_train, y_train)
  acc, my_acc, preds, scores = model.test(clf, X_test, y_test)
  print 'time:', time.time()-begin, 'acc:', acc, 'my_acc:', my_acc
  print 'preds:', preds
  print 'scores:', scores

  print (preds == y_test)[:100]
  print np.count_nonzero(scores > 0)
  print np.count_nonzero(scores < 0)

main()



