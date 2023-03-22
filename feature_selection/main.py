import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from generate_population import *
from model import *
from fitness_score import *

df = pd.read_csv("/home/adminuser/Downloads/sample_house.csv")

df = df[["MoSold", "MSSubClass", "LotFrontage", "LotArea", "MiscVal", "YrSold", "SalePrice", "SaleType", "LotShape"]]

# One-Hot Encoding
sale_type_encoding = pd.get_dummies(df.SaleType)
df = pd.concat([df, sale_type_encoding], axis=1)
lot_shape_encoding = pd.get_dummies(df.LotShape)
df = pd.concat([df, lot_shape_encoding], axis=1)
#sales_condition_encoding = pd.get_dummies(df.SaleCondition)
#df = pd.concat([df, sales_condition_encoding], axis=1)


df.drop(["SaleType", "LotShape",], axis=1, inplace=True)
df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage.mean())

# Splitting X and y
y = df["SalePrice"]
df.drop(["SalePrice"], axis=1, inplace=True)
X = df


train_X, test_X, train_y, test_y = train_test_split(X, y)

n_features = len(train_X.columns)

generator = GenPopulation()

population = generator.generate(n_features, pow(n_features, 2), 0)

model_builder = BuildModel("rf")

svm_model = model_builder.build()

fitness_function = FitnessFunction(
                            population, 
                            svm_model, 
                            train_X,
                            train_y,
                            test_X,
                            test_y,
                            train_X.columns,
                            "rmse"
                        )

fitness_score = fitness_function.get_fitness_score(regression=True)
print(fitness_score)