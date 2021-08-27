import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# TODO: Combine training and testing data input and removing target value.
train = pd.read_csv('train.csv')
labels = train["SalePrice"]
test = pd.read_csv('test.csv')
data = pd.concat([train, test], ignore_index=True, sort=False)
data = data.drop("SalePrice", 1)
ids = test["Id"]

# TODO: Examine nan values in the data.
nans = data.isnull().sum()
print(nans[nans > 0], "\n")


# TODO: Remove id and columns with more than a thousand missing values
data = data.drop("Id", 1)
data = data.drop("Alley", 1)
data = data.drop("Fence", 1)
data = data.drop("MiscFeature", 1)
data = data.drop("PoolQC", 1)
data = data.drop("FireplaceQu", 1)

# TODO: Check data-types of column examine number of categorical and non-categorical features.
print(data.dtypes.value_counts(), "\n")

# all_columns = data.columns.values
# non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
#                    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
#                    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
#                    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
#                    "ScreenPorch","PoolArea", "MiscVal"]
#
# categorical = [value for value in all_columns if value not in non_categorical]
#
# TODO: Get dummy variable for categorical features.
data = pd.get_dummies(data)

# TODO: Fill nan values.
imp = SimpleImputer(strategy='most_frequent')
data = imp.fit_transform(data)

data = np.log(data)
labels = np.log(labels)
data[data == -np.inf] = 0
print(data.shape, "\n")

pca = PCA(whiten=True)  # whitening of data to means to make features less correlated and giving them the same variance.
pca.fit(data)
variance = pd.DataFrame(pca.explained_variance_ratio_)  # Provide variance ratio of each column(column=36)
print(np.cumsum(pca.explained_variance_ratio_), "\n")  # Gives the number of component along which variance is distributed.

pca = PCA(n_components=36, whiten=True)  # Taken n_component=36 because the distribution of variance is only at the first 36.
dataPCA = pca.fit_transform(data)

# print(dataPCA)


def lets_try(train, labels):
    results = {}

    def test_model(clf):
        cv = KFold(n_splits=5, shuffle=True, random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(clf, train, labels, cv=cv, scoring=r2)
        scores = [r2_val_score.mean()]
        return scores

    clf = linear_model.LinearRegression()
    results["Linear"] = test_model(clf)

    clf = linear_model.Ridge()
    results['Ridge'] = test_model(clf)

    clf = linear_model.HuberRegressor()
    results['Huber'] = test_model(clf)

    clf = linear_model.Lasso()
    results['Lasso'] = test_model(clf)

    clf = RandomForestRegressor()
    results["RandomForest"] = test_model(clf)

    clf = AdaBoostRegressor()
    results["AdaBoost"] = test_model(clf)

    clf = svm.SVR()
    results["SVM RBF"] = test_model(clf)

    clf = svm.SVR(kernel="linear")
    results["SVM Linear"] = test_model(clf)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.columns = ["R Square Score"]
    results.plot(kind="bar", title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([-0.1, 1])
    plt.show()
    return results


train = dataPCA[:1460]
test = dataPCA[1460:]

print("Result: \n", lets_try(train, labels))
