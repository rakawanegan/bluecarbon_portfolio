#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:22:35 2023

@author: nakagawa
"""


import pandas as pd
from sklearn.model_selection import train_test_split


class NestedClossValidation():
    def __init__(self,original_df):
        original_x = original_df.drop(["cover"],axis=1)
        original_y = original_df["cover"]
        
        x_presplit1, x_presplit2, y_presplit1, y_presplit2 = train_test_split(original_x, original_y,
                                                            test_size=0.5,
                                                            shuffle=True,
                                                            random_state=314)
        
        
        x_train1, x_train2, y_train1, y_train2 = train_test_split(x_presplit1, y_presplit1,
                                                            test_size=0.5,
                                                            shuffle=True,
                                                            random_state=314)
        
        x_train3, x_train4, y_train3, y_train4 = train_test_split(x_presplit2, y_presplit2,
                                                            test_size=0.5,
                                                            shuffle=True,
                                                            random_state=314)

        self.x_trains = [
                    x_train1,
                    x_train2,
                    x_train3,
                    x_train4
            ]
        self.y_trains = [
                    y_train1,
                    y_train2,
                    y_train3,
                    y_train4     
            ]
        
    def call(self, idx:int=0, evalsize:float=0.3,state=314):
        if not(0<=idx<=3):
            print(f"idx must be integer(0 to 3)\nbut inputs idx is {idx}")
            return False
        x_test = self.x_trains[idx]
        y_test = self.y_trains[idx]
        if idx == 0:
            x_train, x_eval, y_train, y_eval = train_test_split(pd.concat(self.x_trains[idx+1:]),
                                                                pd.concat(self.y_trains[idx+1:]),
                                                                test_size=evalsize,
                                                                shuffle=True,
                                                                random_state=state)
        elif idx == 3:
            x_train, x_eval, y_train, y_eval = train_test_split(pd.concat(self.x_trains[:idx-1]),
                                                                pd.concat(self.y_trains[:idx-1]),
                                                                test_size=evalsize,
                                                                shuffle=True,
                                                                random_state=state)
        else:
            x_train, x_eval, y_train, y_eval = train_test_split(pd.concat(self.x_trains[:idx-1] + self.x_trains[idx:]),
                                                                pd.concat(self.y_trains[:idx-1] + self.y_trains[idx:]),
                                                                test_size=evalsize,
                                                                shuffle=True,
                                                                random_state=state)
        
        return x_train,x_eval,x_test,y_train,y_eval,y_test
   
    

    
if __name__ == "__main__":
    original_df = pd.read_csv("original_data/train_data.csv")
    ncv = NestedClossValidation()
    x_train,x_eval,x_test,y_train,y_eval,y_test = ncv.call(idx=2,evalsize=0.3)
