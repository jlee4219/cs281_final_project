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
  vocab, tweets, authors = read.read_tweets("../training_set_tweets.txt", "../test_set_tweets.txt")
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

  # Create lowbow representation
  begin = time.time()
  count = 0
  for tweet_id in tweets:
    X_all = features.get_features(tweets[tweet_id], vocab_dict, 0.25)
    print X_all.shape
    break
  print "Finished getting features:", time.time() - begin

  clf = model.train(X_train, Y_train)
  acc, my_acc, preds, scores = model.test()
  print 'acc:', acc, 'my_acc:', my_acc
  print 'preds:', preds
  print 'scores:', scores

main()