import numpy as np

# Document frequency for feature selection.
# data is a tweets X feats array
# cutoffs determines feature presence. It is an array of size feats
# num_feats determines the number of features to be selected
def df(data, authors, cutoffs):
  counts = np.zeros(data.shape[1])

  # for each feature, count the number of documents where the feature's above the cutoff
  for feature in range(data.shape[1]):
    counts[feature] = len(np.where(data[:, feature] > cutoffs[feature])[0])

  # return the indices corresponding to the num_feats largest counts
  return counts

# Darmstadt Indexing Approach. Same input as above except for authors
# authors is an array of size tweets providing the corresponding author for each tweet in data
def dia(data, authors, cutoffs):
  author_list = list(set(authors))
  vals = np.zeros((data.shape[1], len(set(authors))))
  for feature in range(data.shape[1]):
    for author_id in range(len(author_list)):
      a = len(np.where(data[np.where(authors == author_list[author_id]), feature]
        > cutoffs[feature])[0])
      b = len(np.where(data[np.where(authors != author_list[author_id]), feature]
        > cutoffs[feature])[0])
      vals[feature, author_id] = float(a) / max(1, (a + b))
  return np.amax(vals, axis=1)

def odds_ratio(data, authors, cutoffs):
  author_list = list(set(authors))
  vals = np.zeros((data.shape[1], len(set(authors))))
  for feature in range(data.shape[1]):
    for author_id in range(len(author_list)):
      a = len(np.where(data[np.where(authors == author_list[author_id]), feature]
        > cutoffs[feature])[0])
      b = len(np.where(data[np.where(authors != author_list[author_id]), feature]
        > cutoffs[feature])[0])
      c = len(np.where(data[np.where(authors == author_list[author_id]), feature]
        <= cutoffs[feature])[0])
      d = len(np.where(data[np.where(authors != author_list[author_id]), feature]
        <= cutoffs[feature])[0])
      vals[feature, author_id] = float(a * d) / max(1, (c * b))
  return np.amax(vals, axis=1)
  # return np.argpartition(vals, - num_feats)[-num_feats:]

# generalized feature selection
# method is a string representing the feature selection 
methods = {'df':df, 'dia':dia, 'odds_ratio':odds_ratio}
def get_scores(selection_data, authors, cutoffs, method):
  idxs = methods[method](selection_data, authors, cutoffs)
  return idxs

def get_indices(scores, num_feats):
  return np.argpartition(scores, -num_feats)[-num_feats:]