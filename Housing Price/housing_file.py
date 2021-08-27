import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Load the data sets.
pd.set_option('display.max_rows', None)
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# TODO: display data set.
# print(df_train.head())
# print("\nRows : ", len(df_train))
# print("\nColumn: ", len(df_train.columns))
# print("\nColumn names : ", df_train.columns)

# TODO: Examine NULL count and removing NULL values.
# print("\n", df_train.isnull().sum())
#
# df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
#
# df_train['Alley'] = df_train['Alley'].fillna('No Alley')
#
# df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean())
#
# df_train['BsmtQual'] = df_train['BsmtQual'].fillna('No basement')
#
# df_train['BsmtCond'] = df_train['BsmtCond'].fillna('No basement')
#
# df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('No basement')
#
# df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('No basement')
#
# df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna('No basement')
#
# df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('No fireplace')
#
# df_train['GarageType'] = df_train['GarageType'].fillna('No garage')
#
# df_train['GarageFinish'] = df_train['GarageFinish'].fillna('No garage')
#
# df_train['GarageQual'] = df_train['GarageQual'].fillna('No garage')
#
# df_train['GarageCond'] = df_train['GarageCond'].fillna('No garage')
#
# df_train['PoolQC'] = df_train['PoolQC'].fillna('No Pool')
#
# df_train['Fence'] = df_train['Fence'].fillna('No fence')
#
# df_train.drop(['MiscFeature'], inplace=True, axis=1)
#
# df_train.dropna(inplace=True)
#
print(df_train.isnull().sum())
print(df_train.shape)

# TODO: Get dummies for columns having categorical data
df_train = pd.get_dummies(df_train, drop_first=True)

# TODO: Remove Columns for which get dummies.
# df_train.drop(categorical, axis=1, inplace=True)

print(df_train.head)

# TODO: Standardization of features
x = df_train.drop(['SalePrice'], axis=1).values
from sklearn.preprocessing import StandardScaler



# pd.set_option('display.max_column', None)
# print(df_train.corr()['LotFrontage'])
# print(df_train.corr()['BsmtQual'])
