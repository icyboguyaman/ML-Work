import pandas as pd
from string import punctuation
from nltk.corpus import stopwords

df = pd.read_csv("SMSSpamCollection.csv", delimiter='\t', names=['label', 'text'])
# print(df.head())

df['Length'] = df['text'].apply(len)
# print(df.head())

sample = "This is a sample message, it doesn't contain punctuation."


def cleaning(sample):
    nonpunc = [c for c in sample if c not in punctuation]
    nonpunc = "".join(nonpunc)
    clean = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    return clean


df['text'] = df['text'].apply(cleaning)
print(df.head())

