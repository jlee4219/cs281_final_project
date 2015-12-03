import numpy as np

# Document frequency for feature selection.
# data is a tweets X feats array
# cutoffs determines feature presence. It is an array of size feats
# num_feats determines the number of features to be selected
def df(data, cutoffs, num_feats):
  counts = np.zeros(data.shape[1])

  # for each feature, count the number of documents where the feature's above the cutoff
  for feature in range(data.shape[1]):
    counts[feature] = len(np.where(data[:, feature] > cutoffs[feature])[0])

  # return the indices corresponding to the num_feats largest counts
  return np.argpartition(counts, -num_feats)[-num_feats:]

# Darmstadt Indexing Approach. Same input as above except for authors
# authors is an array of size tweets providing the corresponding author for each tweet in data
def dia(data, authors, cutoffs, num_feats):
  dia = np.zeros(data.shape[1], len(set(authors)))
  for feature in range(data.shape[1]):
    for author in set(authors):
      a = len(np.where(data[:, np.where(authors == author)] > cutoffs[feature])[0])
      b = len(np.where(data[:, np.where(authors != author)] > cutoffs[feature])[0])
      dia[feature, author] = float(a) / (a + b)
  dia = np.amax(dia, axis=1)