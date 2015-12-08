import numpy as np
from scipy.stats import norm
import string

print 'Imported Features.py'

'''def lowbow(tweet, vocab, sigma):
  ret = np.zeros(len(vocab))
  words = tweet.split()
  for j in range(1, len(words)):
    for word in words:
      s = word.lower().strip()
      s = s.translate(string.maketrans("",""), string.punctuation)
      if s:
        idx = vocab[s]
        ret += kernel(j, sigma, idx, idx + 1)
  return ret

def kernel(mu, sigma, x1, x2):
  C = norm.cdf(1 - mu, sigma) - norm.cdf(-1 * mu, sigma)
  return (norm.cdf(x2, mu, sigma) - norm.cdf(x1, mu, sigma)) / C'''

# bag of words representation
def bow(tweet, vocab_dict):
  ret = np.zeros(len(vocab_dict))
  words = tweet.split()
  for word in words:
    s = word.lower().strip()
    s = s.translate(string.maketrans("",""), string.punctuation)
    if s and s in vocab_dict:
      ret[vocab_dict[s]] += 1
  return ret

def get_bigrams(tweet, bigram_dict):
  ret = np.zeros(len(bigram_dict))
  cleaned_tweet = [word.lower().strip().translate(string.maketrans("",""), string.punctuation) \
                  for word in tweet.split()]
  for i in range(len(cleaned_tweet) - 1):
    s1 = cleaned_tweet[i]
    s2 = cleaned_tweet[i + 1]
    if s1 and s2 and (s1, s2) in bigram_dict:
      ret[bigram_dict[(s1, s2)]] += 1
  return ret

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
  char_counts = map(lambda x: float(x) / max(1, total), char_counts)
  word_counts = map(lambda x: float (x) / max(1, len(tweet.split())), word_counts)
  ret = char_counts + word_counts
  lower = tweet.lower()
  ret = ret + [lower.count("co"), lower.count("me"), lower.count("we")]
  return np.array(ret)

# letters per word, whitespace, uppercase, punctuation, 'ref' 

# tweet is the pure text of the tweet
# vocab is a dictionary from the lowercase words to their ids
# sigma is a parameter for lowbow. We may want to generalize this.
def get_full_feats(tweet, vocab, bigrams):
  return np.hstack((bow(tweet, vocab), get_bigrams(tweet, bigrams), get_counts(tweet)))

# vocab_list representation
def get_num_full_feats(vocab, bigrams):
  return len(vocab) + len(bigrams) + 10