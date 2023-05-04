#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def save_hm(array:pd.DataFrame,title:str)->None:
	plt.figure()
	plt.title(title)
	sns.heatmap(array)
	line_plot = sns.lineplot()
	figure = line_plot.get_figure()
	figure.savefig(title)

columns = list(range(1,13))
index = list(range(1999,2021))
storm = pd.read_csv("okinawa.csv",index_col=0, encoding="shift-jis").drop("年間",axis=1).loc[1999:2020,:]
ymc = pd.read_csv("../../official_data/train_data.csv",index_col=0)[["year","month","cover"]]

cover_max = np.zeros((22,12))
cover_min = np.zeros((22,12))
cover_mean = np.zeros((22,12))
cover_size = np.zeros((22,12))

for i in range(22):
    for j in range(12):
        year=1999+i
        month=1+j
        clist = ymc[(ymc["year"]==year) & (ymc["month"]==month)]["cover"]
        if len(clist)==0:
            pass
        else:
            cover_max[i,j] = clist.max()
            cover_min[i,j] = clist.min()
            cover_mean[i,j] = clist.mean()
            cover_size[i,j] = len(clist)

save_hm(pd.DataFrame(storm.values,index=index,columns=columns),"storm")
save_hm(pd.DataFrame(cover_max,index=index,columns=columns),"cover_max")
save_hm(pd.DataFrame(cover_min,index=index,columns=columns), "cover_min")
save_hm(pd.DataFrame(cover_mean,index=index,columns=columns), "cover_mean")
save_hm(pd.DataFrame(cover_size,index=index,columns=columns), "cover_size")