import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import clean_text

def load_train_test_imdb_data(data_dir):
	"""
	Loads IMDB data from a folder path

	Input: 
	data_dir: path to aclImdb folder

	Output: train/test datasets as pd dataframes
	"""

	data = {}
	for split in ["train", "test"]:
		data[split] = []
		for sentiment in ["neg", "pos"]:
			score = 1 if sentiment == "pos" else 0
			path = os.path.join(data_dir, split, sentiment)
			filenames = os.listdir(path)
			for filename in filenames:
				with open(os.path.join(path, filename),"r") as f:
					review = f.read()
					data[split].append([review, score])

	np.random.shuffle(data["train"])
	data["train"] = pd.DataFrame(data["train"], columns=["text", "sentiment"])

	np.random.shuffle(data["test"])
	data["test"] = pd.DataFrame(data["test"], columns=["text", "sentiment"])

	return data["train"], data["test"]


####### Main #######

train_data, test_data = load_train_test_imdb_data("../../aclImdb")

vectorizer = TfidfVectorizer(stop_words = "english", preprocessor = clean_text, ngram_range=(1,2))
train_features = vectorizer.fit_transform(train_data["text"])
test_features = vectorizer.transform(test_data["text"])

model = LinearSVC()
model.fit(train_features, train_data["sentiment"])
ypred = model.predict(test_features)
accuracy = accuracy_score(test_data["sentiment"], ypred)
print("Accuracy of text classification for imdb dataset: {:.2f}".format(accuracy*100))				#Obtained accuracy = 88.66%

ypred = model.predict(train_features)
accuracy = accuracy_score(train_data["sentiment"], ypred)
print("Accuracy of text classification for imdb dataset: {:.2f}".format(accuracy*100))				#Obtained accuracy = 99.99%