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

	# to generate vocabulary, strip spaces and punctuation and make lowercase
	for author in train_word:

		for tweet in train_word[author]:
			for word in tweet.split():
				s = word.lower().strip()
				s = s.translate(string.maketrans("",""), string.punctuation)
				if s:
					word_counts[s] += 1
	
	for word in word_counts:
		if word_counts[word] > 1:
			vocab.add(word)

	return vocab, train_word, test_word, train_char, test_char