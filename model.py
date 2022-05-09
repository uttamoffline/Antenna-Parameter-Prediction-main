import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pickle
df = pd.read_csv(r"dataset_antenna.csv")
train_Y = df["length of patch in mm"]
train_w =df["width of patch in mm"]
train_X = df.drop("length of patch in mm" ,axis = 1)
train_X = (train_X.drop("width of patch in mm" , axis = 1))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# train_X , test_X , train_Y , test_Y = train_test_split(train_X , train_l, random_state = 4 , test_size= 0.3)
# sc = StandardScaler()
# train_X = sc.fit_transform(train_X)
# test_X = sc.fit_transform(test_X)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5 , include_bias= False)
train_X = poly.fit_transform(train_X)
from sklearn.linear_model import LinearRegression
lr  = LinearRegression()
lr.fit(train_X,train_Y)
lr1 = LinearRegression()
lr1.fit(train_X , train_w)
pickle.dump(lr,open("model.pkl","wb"))
pickle.dump(lr1,open("model1.pkl","wb"))
model = pickle.load(open("model.pkl","rb"))
model1 = pickle.load(open("model1.pkl","rb"))

