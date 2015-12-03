import os
import time
import string
import numpy as np
from sklearn import cross_validation

def get_tweet(line, authors, vocab, seen_twice, count):
  line = line.split("\t")
  if len(line) < 4:
    count += 1
    return count
  try:
    tweet = " ".join(line[2:-1])
    if not tweet:
      return count
    if int(line[0]) in authors:
      authors[int(line[0])].append(tweet)
    else:
      authors[int(line[0])] = [tweet]

    # to generate vocabulary, strip spaces and punctuation and make lowercase
    for word in tweet.split():
      s = word.lower().strip()
      s = s.translate(string.maketrans("",""), string.punctuation)
      if s:
        if s in vocab:
          seen_twice.add(s)
        vocab.add(s)
  except ValueError:
    count += 1
  return count

def read_tweets(train_file, test_file):
  train = open(train_file, 'r')
  test = open(test_file, 'r')
  vocab = set()
  seen_twice = set()
  authors = {}
  lines = 0
  count = 0
  for line in train:
    count = get_tweet(line, authors, vocab, seen_twice, count)
    lines += 1
  for line in test:
    count = get_tweet(line, authors, vocab, seen_twice, count)
    lines += 1

  best_authors = sorted([(author, len(authors[author])) for author in authors], key=lambda x:x[1], reverse=True)
  best_authors = [x[0] for x in best_authors]
  for author in authors:
    if not author in best_authors:
      del authors[author]
  train_data, test_data = split_train_test(authors)

  train.close()
  test.close()
  return seen_twice, train_data, test_data

def split_train_test(authors):
  train = {}
  test = {}
  for author in authors:
    for tr, te in cross_validation.ShuffleSplit(len(authors[author]), 1, 0.05):
      train[author] = np.array(authors[author])[tr]
      test[author] = np.array(authors[author])[te]
  return train, test
