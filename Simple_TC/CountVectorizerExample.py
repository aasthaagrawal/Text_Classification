import pandas as pd

from preprocessing import clean_text

from sklearn.feature_extraction.text import CountVectorizer

training_text = ["This is a good cat.", "This is a bad day."]
test_text = ["This day is a good day."]

vectorizer = CountVectorizer(stop_words = "english", preprocessor = clean_text)
vectorizer.fit(training_text)
print(vectorizer.transform(training_text))

#get vocabulary of the vectorizer
inv_vocab = {v: k for k,v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
print(vocabulary)

print(test_text)
print(pd.DataFrame(
    data=vectorizer.transform(test_text).toarray(),
    index=["test sentence"],
    columns=vocabulary
))