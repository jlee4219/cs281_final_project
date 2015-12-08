import os
import time
import string
import numpy as np
from sklearn import cross_validation

print 'Imported read.py'

def get_tweet(line, authors, vocab, count):
  line = line.split("\t")
  if len(line) < 4:
    count += 1
    return count
  try:
    #line is userID, tweetID, tweet, date
    tweet = " ".join(line[2:-1])
    author_id = int(line[0])
    if not tweet:
      return count
    if author_id in authors:
      authors[author_id].append(tweet)
    else:
      authors[author_id] = [tweet]

  except ValueError:
    count += 1
  return count

def read_tweets(train_file, test_file):
  train = open(train_file, 'r')
  test = open(test_file, 'r')
  vocab = {}
  authors = {}
  lines = 0
  count = 0
  start = time.time()
  for line in train:
    count = get_tweet(line, authors, vocab, count)
    if lines % 100000 == 0:
      start = time.time()
    lines += 1
  for line in test:
    count = get_tweet(line, authors, vocab, count)
    if lines % 100000 == 0:
      start = time.time()
    lines += 1

  best_authors = sorted([(author, len(authors[author])) for author in authors], key=lambda x:x[1], reverse=True)
  best_authors = [x[0] for x in best_authors]
  best_authors = best_authors[:20]
  authors = {k:v for (k,v) in authors.iteritems() if k in best_authors}

  train_data, test_data = split_train_test(authors)

  train.close()
  test.close()
  return vocab, train_data, test_data

def split_train_test(authors):
  train = {}
  test = {}
  for author in authors:
    for tr, te in cross_validation.ShuffleSplit(len(authors[author]), 1, 0.05):
      train[author] = np.array(authors[author])[tr]
      test[author] = np.array(authors[author])[te]
  return train, test
