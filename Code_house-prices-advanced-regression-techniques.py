import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# path file and create dataset(Train)
file_train="D:\\Books\\project\\0002-House Prices - Advanced Regression Techniques\\Train_new.csv"

dataset = pd.read_csv(file_train)

if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop('Unnamed: 0', axis=1)

# fill NaN and create good data (Train)
dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(0)

dataset['BsmtQual']=dataset['BsmtQual'].fillna('No')

dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('No')

dataset['LotFrontage']=dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

dataset['ExterQual']=dataset['ExterQual'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 })

dataset['KitchenQual']=dataset['KitchenQual'].replace({'Ex': 4,  'Gd': 3,  'TA': 2,  'Fa':1})

dataset['BsmtQual']=dataset['BsmtQual'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 ,'No':0 })

dataset['FireplaceQu']=dataset['FireplaceQu'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 , 'No':0})

dataset['Neighborhood']=LabelEncoder().fit_transform(dataset['Neighborhood'])

dataset['CentralAir']=dataset['CentralAir'].replace({'Y':1 , 'N':0})

# train and make Model(train)
x=dataset.drop('SalePrice',axis=1)

y=dataset['SalePrice']

model=make_pipeline(StandardScaler(),XGBRegressor()).fit(x,y)

print(r2_score(model.predict(x),y))

# path file and create dataset(Test)
file_train="D:\\Books\\project\\0002-House Prices - Advanced Regression Techniques\\test.csv"

dataset_test = pd.read_csv(file_train)

test_ids = dataset_test['Id']

dataset_test=dataset_test[['OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageCars','GarageArea','YearBuilt','YearRemodAdd',
                            'MasVnrArea','ExterQual','KitchenQual','BsmtQual','FireplaceQu','TotRmsAbvGrd','BedroomAbvGr','FullBath','HalfBath',
                        	'Neighborhood','LotFrontage','OpenPorchSF','WoodDeckSF','PoolArea','CentralAir']]

# fill NaN and create good data (Test)
dataset_test['MasVnrArea']=dataset_test['MasVnrArea'].fillna(0)

dataset_test['BsmtQual']=dataset_test['BsmtQual'].fillna('No')

dataset_test['FireplaceQu'] = dataset_test['FireplaceQu'].fillna('No')

dataset_test['LotFrontage']=dataset_test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
dataset_test['LotFrontage']=dataset_test['LotFrontage'].fillna(dataset_test['LotFrontage'].mean())

dataset_test['ExterQual']=dataset_test['ExterQual'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 })

dataset_test['KitchenQual']=dataset_test['KitchenQual'].replace({'Ex': 4,  'Gd': 3,  'TA': 2,  'Fa':1})

dataset_test['BsmtQual']=dataset_test['BsmtQual'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 ,'No':0 })

dataset_test['FireplaceQu']=dataset_test['FireplaceQu'].replace({'Ex': 5,  'Gd': 4,  'TA': 3,  'Fa': 2,  'Po': 1 , 'No':0})

dataset_test['Neighborhood']=LabelEncoder().fit_transform(dataset_test['Neighborhood'])

dataset_test['CentralAir']=dataset_test['CentralAir'].replace({'Y':1 , 'N':0})

# train and make Model(Test)
X_test=dataset_test

predict = model.predict(X_test)

# plot
plt.scatter(model.predict(x),y)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.show()

# Finall 
print(dataset_test.columns.tolist())
data_Finall=pd.DataFrame({'Id':test_ids,'SalePrice':predict})
data_Finall.to_csv('D:\\Books\\project\\0002-House Prices - Advanced Regression Techniques\\Finall.csv',index=False)