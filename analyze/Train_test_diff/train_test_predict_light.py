#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:43:55 2023

@author: nakagawa
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


original_train_df = pd.read_csv("../../official_data/train_data.csv",index_col=0).drop(["cover",'YMD','Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1)
original_test_df = pd.read_csv("../../official_data/test_data.csv",index_col=0).drop(['YMD','Landsat_StartTime','PRODUCT_ID','mesh20'],axis=1)

train_df = pd.concat([original_train_df,pd.Series([0]*len(original_train_df)).rename("traintest")],axis=1)
test_df = pd.concat([original_test_df,pd.Series([1]*len(original_test_df)).rename("traintest")],axis=1)

df = pd.concat([train_df,test_df],axis=0)

scorelist = []
bestparamslist = []

x_traineval,x_test,y_traineval,y_test = train_test_split(df.drop("traintest",axis=1),
                                                         df["traintest"],
                                                         test_size=0.1,
                                                         random_state=314)

x_train,x_eval,y_train,y_eval = train_test_split(x_traineval,
                                                 y_traineval,
                                                 test_size=0.3,
                                                 random_state=314)


lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x_eval, y_eval, reference=lgb_train, free_raw_data=False)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
}


model = lgb.train(params,                              
                  lgb_train,                           
                  num_boost_round=1000,                
                  valid_names=['train', 'valid'],      
                  valid_sets=[lgb_train, lgb_eval],    
                  verbose_eval=-1                      
              )
pred = pd.Series(model.predict(x_test)).map(round)

count=0
misslist=[]
for i in range(len(pred)):
    if pred[i] == y_test.reset_index().drop("index",axis=1)["traintest"][i]:
        count+=1
    else:
        misslist.append(y_test.reset_index()["index"][i])
print(count/len(pred))
#0.993 allparam


importance = pd.DataFrame(model.feature_importance(importance_type = "gain")
                          , index=x_train.columns, columns=['gain_importance'])
mostgain_importance = importance.sort_values("gain_importance",
                                              ascending=False)/importance.sum()
mostgain_importance.to_csv("most_gain_importance.csv")