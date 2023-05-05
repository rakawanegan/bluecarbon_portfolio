#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:42:38 2023
coverを消すコードを実装する
@author: nakagawa
"""

import os
import pandas as pd

def mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

mkdir("split_train")
mkdir("split_test")
mkdir("split_train/landsatperyear")
mkdir("split_test/landsatperyear")
mkdir("submits")

df_train = pd.read_csv("original_data/train_data.csv")
df_test = pd.read_csv("original_data/test_data.csv")


for i in range(21):
    df_part = df_train.filter(like=f"20{str(i).zfill(2)}")    
    df_part.to_csv(f'split_train/landsatperyear/landsatperyaear_20{str(i).zfill(2)}_traindata.csv')

    df_part = df_test.filter(like=f"20{str(i).zfill(2)}")    
    df_part.to_csv(f'split_test/landsatperyear/landsatperyaear_20{str(i).zfill(2)}_testdata.csv')

dic = {
        "被度文献データ": "_literal",
        "海洋環境要因データ": "_seaenviroment",
        "時系列「ランドサット」衛星画像データ": "landsat_series",
        "2019年「センチネル2」衛星画像データ": "sentinel2019",
        "年ごとの「ランドサット」衛星画像データ": "landsat_peryear",
        "グリッド区分": "grid_division"
       }

col_df_train=pd.read_excel('original_data/feature_description.xlsx')
col_df_train=col_df_train[~col_df_train.iloc[:,0].isin(['SAVImir'])]
for c in col_df_train.iloc[:,1].unique():
    colname_train=col_df_train[col_df_train.iloc[:,1].isin([c])].iloc[:,0].array
    df_train[colname_train].to_csv(f'split_train/{dic[c]}.csv')
    
col_df_test=pd.read_excel('original_data/feature_description_test.xlsx')
col_df_test=col_df_test[~col_df_test.iloc[:,0].isin(['SAVImir'])]

for c in col_df_test.iloc[:,1].unique():
    colname_test=col_df_test[col_df_test.iloc[:,1].isin([c])].iloc[:,0].array
    df_test[colname_test].to_csv(f'split_test/{dic[c]}.csv')
