import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

# TODO: Analyse NULL values.
# print(df.isnull().sum())

# TODO: Analyse co-relation between independent variable.
# sns.pairplot(df[["age", "trestbps", "chol", "thalach", "oldpeak"]], kind='reg')
# sns.heatmap(df.corr(), annot=True)
# plt.show()
# print(df['age'].value_counts()/df[''])
# sns.countplot(x='sex', data=df, hue='target')
# sns.countplot(x='age', data=df, hue='sex')
# pd.crosstab(df.age, df.target).plot(kind="bar")
# plt.show()


# TODO: Replace numerical categorical values by different values.

df['cp'] = df['cp'].replace(0, 'cp_0')
df['cp'] = df['cp'].replace(1, 'cp_1')
df['cp'] = df['cp'].replace(2, 'cp_2')
df['cp'] = df['cp'].replace(3, 'cp_3')

df['slope'] = df['slope'].replace(0, 'slope_0')
df['slope'] = df['slope'].replace(1, 'slope_1')
df['slope'] = df['slope'].replace(2, 'slope_2')

df['ca'] = df['ca'].replace(0, "ca_0")
df['ca'] = df['ca'].replace(1, "ca_1")
df['ca'] = df['ca'].replace(2, "ca_2")
df['ca'] = df['ca'].replace(3, "ca_3")


# TODO: Create dummy variables and data cleaning.
cp = pd.get_dummies(df['cp'], drop_first=True)
slope = pd.get_dummies(df['slope'], drop_first=True)
ca = pd.get_dummies(df['ca'], drop_first=True)


df = pd.concat([df, cp, slope, ca], axis=1)
df.drop(["cp", "slope", "ca"], axis=1, inplace=True)

# print(df.head())

# TODO: Data Selection.
x = df.drop(['target', "restecg", "thal"], axis=1)
y = df["target"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# TODO: Logistic Regression.
from sklearn.linear_model import LogisticRegression
obj = LogisticRegression()
obj.fit(x_train, y_train)
y_predict = obj.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
