import features
import read
import time
import numpy as np

def main():
  # Read the data from the text files
  begin = time.time()
  vocab, tweets, authors = read.read_tweets("../training_set_tweets.txt", "../test_set_tweets.txt")
  print "Read data:", time.time() - begin

  # Assign ids to words
  begin = time.time()
  vocab_list = list(vocab)
  vocab_list.sort()
  vocab_dict = {}
  for i in range(len(vocab_list)):
    vocab_dict[vocab_list[i]] = i
  print "Assigned ids to words:", time.time() - begin

  # Create lowbow representation
  begin = time.time()
  count = 0
  for tweet_id in tweets:
    a = features.get_features(tweets[tweet_id], vocab_dict, 0.25)
    print a.shape
    break
  print "Finished getting features:", time.time() - begin

main()