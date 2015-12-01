import numpy as np
from scipy.stats import norm

def lowbow(tweet, vocab, sigma):
  ret = np.zeros(len(vocab))
  words = tweet.split()
  for j in len(1, words):
    for i in len(words):
      s = word.lower().strip()
      s = s.translate(string.maketrans("",""), string.punctuation)
      idx = vocab[s]
      ret += kernel(j, sigma, idx, idx + 1)
  return ret

def kernel(mu, sigma, x1, x2):
  C = norm.cdf(1 - mu, sigma) - norm.cdf(-1 * mu, sigma)
  return (norm.cdf(x2, mu, sigma) - norm.cdf(x1, mu, sigma)) / C

# tweet is the pure text of the tweet
# vocab is a dictionary from the lowercase words to their ids
# sigma is a parameter for lowbow. We may want to generalize this.
def get_features(tweet, vocab, sigma):
  return lowbow(tweet, vocab, sigma)