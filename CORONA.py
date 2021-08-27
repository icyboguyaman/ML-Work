import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("corona.csv")

sns.heatmap(df.corr(), annot=True)
plt.show()

x = df[['Recovered']]
y = df[['Deaths']]

from sklearn.model_selection import train_test_split
