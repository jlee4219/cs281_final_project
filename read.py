import os
import time
import string

def get_tweet(line, tweets, authors, vocab, count):
  line = line.split("\t")
  if len(line) < 4:
    count += 1
    return count
  try:
    tweet = " ".join(line[2:-1])
    tweets[int(line[1])] = tweet
    authors[int(line[1])] = int(line[0])

    # to generate vocabulary, strip spaces and punctuation and make lowercase
    for word in tweet.split():
      s = word.lower().strip()
      s = s.translate(string.maketrans("",""), string.punctuation)
      if s:
        vocab.add(s)
  except ValueError:
    count += 1
  return count

def read_tweets(train_file, test_file):
  train = open(train_file, 'r')
  test = open(test_file, 'r')
  vocab = set()
  tweets = {}
  authors = {}
  lines = 0
  count = 0
  for line in train:
    count = get_tweet(line, tweets, authors, vocab, count)
    lines += 1
  print "Skipped:", float(count)/lines
  # get_tweets(data, tweets, authors, vocab)
  return vocab, tweets, authors
  train.close()
  test.close()

# test code, no need to actually execute
begin = time.time()
vocab, tweets, authors = read_tweets("training_set_tweets.txt", "test_set_tweets.txt")
vocab_file = open("twitter_train_vocab.txt", 'w')
vocab_list = list(vocab)
vocab_list.sort()
for i in vocab_list:
  print i
  vocab_file.write(i)
vocab_file.close()
print len(vocab), len(tweets), len(authors)
print "Time:", time.time() - begin