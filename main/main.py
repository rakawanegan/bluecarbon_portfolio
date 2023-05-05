import sys
sys.dont_write_bytecode = True

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from preprocessor.data_fill import MedianIndexFull as MIF
from preprocessor.read_info import Pre_Process_Randsat,Fillnan_Randsat_Series
from sklearn.model_selection import train_test_split

from predictor.kNearestNeighbor import kNearestNeighbor as kNN
from predictor.NeuralNetwork import NeuralNetwork as NN
from predictor.RandomForest import RandomForest as RF
from predictor.LightGBMoptuna import LightGBM as LGBM

from evaluator.evaluate import Output


train_df = pd.read_csv("../official_data/train_data.csv",index_col=0)
test = pd.read_csv("../official_data/test_data.csv",index_col=0)

X = train_df.drop("cover",axis=1)
Y = train_df["cover"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,shuffle=True,random_state=314)


# preprocess
print("landsat 2000-2020 filling...")
pp=Pre_Process_Randsat()
x_train = pp.fit_transform(x_train)
x_test = pp.transform(x_test)
test = pp.transform(test)


print("landsat series filling...")
frs = Fillnan_Randsat_Series()
x_train = frs.fit_transform(x_train,'中央値')
x_test = frs.transform(x_test)
test = frs.transform(test)

print("all missing value filling...")
x_train = x_train.drop(['YMD','Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1)
x_test = x_test.drop(['YMD','Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1)
test = test.drop(['YMD','Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1)
mif = MIF()
mif.fit(x_train)
x_train = mif.transform(x_train)
x_test = mif.transform(x_test)
test = mif.transform(test)

print("minmax scaling...")
mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)
test = mms.transform(test)

print("standard scaling...")
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
test = ss.transform(test)

x_train = pd.DataFrame(x_train,columns=train_df.drop(['YMD',"cover",'Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1).columns).iloc[:,:84]
x_test = pd.DataFrame(x_test,columns=train_df.drop(['YMD',"cover",'Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1).columns).iloc[:,:84]
test = pd.DataFrame(test,columns=train_df.drop(['YMD',"cover",'Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1).columns).iloc[:,:84]


# fit and predict
print("initiating...")
knn = kNN()
nn = NN(x_train)
rf = RF()
lgbm = LGBM()

print("learning...")
knn.fit(x_train,y_train)
nn.fit(x_train,y_train)
rf.fit(x_train,y_train)
lgbm.fit(x_train,y_train)

print("predicting...")
knn_predict = knn.predict(x_test)
nn_predict = nn.predict(x_test)
rf_predict = rf.predict(x_test)
lgbm_predict = lgbm.predict(x_test)

knn_submit = knn.predict(test)
nn_submit = nn.predict(test)
rf_submit = rf.predict(test)
lgbm_submit = lgbm.predict(test)

y_test = y_test.reset_index()
print("ensambling...")
y_predict = pd.DataFrame(pd.concat([knn_predict['prediction'],
                                  nn_predict['prediction'],
                                  rf_predict['prediction'],
                                  lgbm_predict['prediction']],axis=1).mean(axis=1),columns=['prediction'])
                                  
submit = pd.DataFrame(pd.concat([knn_submit['prediction'],
                                 nn_submit['prediction'], 
                                 rf_submit['prediction'], 
                                 lgbm_submit['prediction']],axis=1).mean(axis=1),index=test.index,columns=['prediction'])


print("evaluating...")
rmse_score = Output.make_score_rmse(y_predict,y_test)

print("submitting...")
Output.submit(submit, f"thistissample_rmse{rmse_score}")

print("run completed!")
