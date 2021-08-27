import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

messages = pd.read_csv("SMSSpamCollection.csv", sep="\t", names=['label', 'message'])

messages['length'] = messages['message'].apply(len)

import string 

from nltk.corpus import stopwords

def text_process(mess):
	nonpunc = [c for c in mess if c not in string.punctuation]
	nonpunc = "".join(nonpunc)
	clean_mess = [word for word in nonpunc.split() if word.lower() not in stopwords.words("english")]
	return clean_mess


from sklearn.feature_extraction.text import CountVectorizer

bow_tranformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# print(len(bow_tranformer.vocabulary_))
# print(bow_tranformer.vocabulary_)

mesg4 = messages['message'][3]
bw4 = bow_tranformer.transform([mesg4])
print(bw4)
print(bw4.shape)

# print(bow_tranformer.get_feature_names()[4068])
message_bow = bow_tranformer.transform(messages['message'])
print(message_bow.nnz)

from sklearn.feature_extraction.text import TfidfTransformer
tfifd_trasfomer = TfidfTransformer().fit(message_bow)
tfifd4 = tfifd_trasfomer.transform(bw4)

print(tfifd4)
print(tfifd_trasfomer.idf_[bow_tranformer.vocabulary_['U']])

messages_tfidf = tfifd_trasfomer.transform(message_bow)
print(messages_tfidf.shape)

x = messages_tfidf
y = messages['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=15)

nb = MultinomialNB()

print("Classification report:-")
nb.fit(x, y)
y_predict = nb.predict(x)
print(classification_report(y, y_predict))

# print("Classification report (training and testing different:-")
# nb.fit(x_train, y_train)
# y_predict = nb.predict(x_test)
# print(classification_report(y_test, y_predict))
