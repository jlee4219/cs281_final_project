import numpy as np
from scipy.stats import norm
import string

def lowbow(tweet, vocab, sigma):
  ret = np.zeros(len(vocab))
  words = tweet.split()
  for j in range(1, len(words)):
    for word in words:
      s = word.lower().strip()
      s = s.translate(string.maketrans("",""), string.punctuation)
      idx = vocab[s]
      ret += kernel(j, sigma, idx, idx + 1)
  return ret

def kernel(mu, sigma, x1, x2):
  C = norm.cdf(1 - mu, sigma) - norm.cdf(-1 * mu, sigma)
  return (norm.cdf(x2, mu, sigma) - norm.cdf(x1, mu, sigma)) / C

# Create all count based features
def get_counts(tweet):
  punctuation = 0
  uppercase = 0
  letters = 0
  whitespace = 0
  digits = 0
  total = 0
  for c in tweet:
    if c in string.punctuation:
      punctuation += 1
    if c.isupper():
      uppercase += 1
    if c.isalpha():
      letters += 1
    if c.isspace():
      whitespace += 1
    if c.isdigit():
      digits += 1
    total += 1
  char_counts = [punctuation, uppercase, whitespace, digits]
  word_counts = [tweet.count("\'"), tweet.count("#"), letters]
  char_counts = map(lambda x: float(x) / total, char_counts)
  word_counts = map(lambda x: float (x) / len(tweet.split()), word_counts)
  ret = char_counts + word_counts
  lower = tweet.lower()
  ret = ret + [lower.count("co"), lower.count("me"), lower.count("we")]
  return np.array(ret)

# tweet is the pure text of the tweet
# vocab is a dictionary from the lowercase words to their ids
# sigma is a parameter for lowbow. We may want to generalize this.
def get_features(tweet, vocab, sigma):
  return np.hstack((lowbow(tweet, vocab, sigma), get_counts(tweet)))