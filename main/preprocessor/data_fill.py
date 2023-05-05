#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


class MedianIndexFull():
     def __init__(self,):
         self.medianlist = list()
            
     def fit(self,x_train:pd.DataFrame):
         for column in x_train:
             self.medianlist.append(x_train[column].median())
            
     def transform(self,x_df_original:pd.DataFrame):
         x_df = x_df_original.copy()
         for idx,column in enumerate(x_df):
             x_df[column] = x_df[column].fillna(self.medianlist[idx])
         return x_df
               
     
class MeanIndexFull():
    def __init__(self):
        self.meanlist = list()
        
    def fit(self, x_train:pd.DataFrame):
        for column in x_train:
              self.meanlist.append(x_train[column].mean())
        
    def transform(self, x_df_original:pd.DataFrame):
        x_df = x_df_original.copy()
        for idx,column in enumerate(x_df):
              x_df[column] = x_df[column].fillna(self.meanlist[idx])
        return x_df
      
