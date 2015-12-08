import string
import re
import copy
from collections import defaultdict

print 'Imported preprocessing.py'


def replaceThings(train, test):
	reURL = re.compile(r'http\S+')
	reREF = re.compile(r'@\S+')
	reREP = re.compile(r'(RT:)|(RT)')
	#01/02/94
	reDATE = re.compile(r'(\d){1-2}/(\d){1-2}/(\d){2-4}')
	#10:24
	reTIME = re.compile(r'(\d){1-2}:(\d){1-2}')
	reNUM = re.compile(r'(\d)+')
	for author in train:
		tweets = train[author]
		for i in xrange(len(tweets)):
			tweets[i] = reURL.sub('<URL>', tweets[i])
			tweets[i] = reREF.sub('<REF>', tweets[i])
			tweets[i] = reREP.sub('<REPLY>', tweets[i])
			tweets[i] = reDATE.sub('<DATE>', tweets[i])
			tweets[i] = reTIME.sub('<TIME>', tweets[i])
			tweets[i] = reNUM.sub('<NUM>', tweets[i]) #replace after dates/times
	return train, test

def removeThings(train, test):
	reURL = re.compile(r'http\S+')
	reREF = re.compile(r'@\S+')
	reREP = re.compile(r'(RT:)|(RT)')
	#01/02/94
	reDATE = re.compile(r'(\d){1-2}/(\d){1-2}/(\d){2-4}')
	#10:24
	reTIME = re.compile(r'(\d){1-2}:(\d){1-2}')
	reNUM = re.compile(r'(\d)+')
	for author in train:
		tweets = train[author]
		for i in xrange(len(tweets)):
			tweets[i] = reURL.sub('', tweets[i])
			tweets[i] = reREF.sub('', tweets[i])
			tweets[i] = reREP.sub('', tweets[i])
			tweets[i] = reDATE.sub('', tweets[i])
			tweets[i] = reTIME.sub('', tweets[i])
			tweets[i] = reNUM.sub('', tweets[i]) #replace after dates/times
	return train, test

def preprocess(train, test):
	train_word = copy.deepcopy(train)
	test_word = copy.deepcopy(test)
	train_char = copy.deepcopy(train)
	test_char = copy.deepcopy(test)
	replaceThings(train_word, test_word)
	removeThings(train_char, test_char)

	word_counts = defaultdict(int)
	word_users = defaultdict(int)
	vocab = set()
	bigrams = set()
	usage = {}
	bigram_usage = {}

	# to generate vocabulary, strip spaces and punctuation and make lowercase
	for author in train_word:
		for tweet in train_word[author]:
			# to generate vocabulary, strip spaces and punctuation and make lowercase
			cleaned_tweet = [word.lower().strip().translate(string.maketrans("",""), string.punctuation) for word in tweet.split()]
			for i in range(len(cleaned_tweet)):
				s = cleaned_tweet[i]
				if s:
					word_counts[s] += 1
					if s in usage:
						usage[s].add(author)
					else:
						usage[s] = set([author])
					if i < len(cleaned_tweet) - 1:
						s2 = cleaned_tweet[i + 1]
						if (s, s2) in bigram_usage:
							bigram_usage[(s, s2)].add(author)
						else:
							bigram_usage[(s, s2)] = set([author])
	
	counter = 0
	used_once = 0
	for word in word_counts:
		if word_counts[word] > 1 and len(usage[word]) > 1:
			vocab.add(word)
		if word_counts[word] > 1 and len(usage[word]) == 1:
			counter += 1
		if word_counts[word] <= 1:
			used_once += 1
	print "Words used in total:", len(word_counts)
	print "Words used >1 by unique user:", counter
	print "Words used by multiple users", len(vocab)
	print "Words used once", used_once

	num_bigrams = 0
	num_chosen = 0
	for bigram in bigram_usage:
		if len(bigram_usage[bigram]) > 1:
			bigrams.add(bigram)
			num_chosen += 1
		num_bigrams += 1

	print num_chosen, "bigrams chosen out of", num_bigrams

	return vocab, bigrams, train_word, test_word, train_char, test_char