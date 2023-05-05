#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:56:22 2023

@author: naoya
"""
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import numpy as np



class Output():     
     def make_score_rmse(y_predict:pd.DataFrame, y_test:pd.DataFrame) -> float:
          result = pd.concat([y_predict, y_test],axis=1)
          rmse_score = np.sqrt(mse(result['prediction'],result['cover']))
          print("rmes score:",rmse_score)
          return rmse_score
          
     def submit(output:pd.DataFrame, filename:str) -> None:
         output.to_csv(f"submits/{filename}.csv",header=False)
         print("output!")
