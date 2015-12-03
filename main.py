import features
import read
import model
import selection

import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def main():
  # Read the data from the text files
  begin = time.time()
  vocab, train, test = read.read_tweets("../training_set_tweets.txt", "../test_set_tweets.txt")
  print "Train size:", len(train), "Test size:", len(test)
  print "Read data:", time.time() - begin

  # Assign ids to words
  begin = time.time()
  vocab_list = list(vocab)
  vocab_list.sort()
  #vocab_dict = dict(zip(vocab_list, range(len(vocab_list)) ))
  vocab_dict = {}
  for i in range(len(vocab_list)):
    vocab_dict[vocab_list[i]] = i
  print "Assigned ids to words:", time.time() - begin

  # Build train and test set
  num_full_feats = len(vocab_list) + 10
  num_train_tweets = 0
  num_test_tweets = 0
  for author_id in train:
    num_train_tweets += len(train[author_id])
  for author_id in test:
    num_test_tweets += len(test[author_id])
  X_train = np.zeros(num_train_tweets, num_full_feats)
  y_train = np.zeros(num_train_tweets)
  X_test = np.zeros(num_test_tweets, num_full_feats)
  y_test = np.zeros(num_test_tweets)

  count = 0

  for author_id in train:
    for tweet in train[author_id]:
      X_train[count, :] = get_full_feats(tweet, vocab)
      y_train[count, :] = author_id
      count += 1

  count = 0
  for author_id in test:
    for tweet in test[author_id]:
      X_test[count, :] = get_full_feats(tweet, vocab)
      y_test[count, :] = author_id
      count += 1


  clf = model.train(X_train, Y_train)
  acc, my_acc, preds, scores = model.test()
  print 'acc:', acc, 'my_acc:', my_acc
  print 'preds:', preds
  print 'scores:', scores


main()



