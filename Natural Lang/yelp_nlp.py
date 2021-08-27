import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv("yelp.xls", sep=',')
print(df.head())
print(df.columns)
print(df['stars'].value_counts())

df.drop(['business_id', 'date', 'review_id', 'type', 'user_id'], axis=1, inplace=True)
df['lenght'] = df['text'].apply(len)

def text_process(sample):
    nonpunc = [c for c in sample if c not in string.punctuation]
    nonpunc = "".join(nonpunc)
    clean_mess = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    return clean_mess


bow = CountVectorizer(analyzer=text_process).fit_transform(df['text'])
tfidf = TfidfTransformer().fit_transform(bow)

x = tfidf
y = df['stars']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

nb = MultinomialNB()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)

print(classification_report(y_predict, y_test))
