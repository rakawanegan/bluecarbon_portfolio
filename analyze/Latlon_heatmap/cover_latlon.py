#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def save_hm(array:np.array,title:str)->None:
	plt.figure()
	plt.title(title)
	sns.heatmap(array)
	line_plot = sns.lineplot()
	figure = line_plot.get_figure()
	figure.savefig(title)
	
	
size = 10

lat = pd.read_csv("../../official_data/train_data.csv",usecols=["lat"])
lon = pd.read_csv("../../official_data/train_data.csv",usecols=["lon"])
latloncover = pd.read_csv("../../official_data/train_data.csv",usecols=["lat","lon","cover"])

lat = lat.applymap(lambda x: size*x).applymap(int)
lon = lon.applymap(lambda x: size*x).applymap(int)
lat_range = lat.max().item()-lat.min().item()
lon_range = lon.max().item()-lon.min().item()

cover_max = np.zeros((lat_range,lon_range))
cover_min = np.zeros((lat_range,lon_range))
cover_mean = np.zeros((lat_range,lon_range))
cover_size = np.zeros((lat_range,lon_range))

for i in range(lat_range):
    for j in range(lon_range):
        clist = latloncover[(lat["lat"]==i+lat.min().item()) & (lon["lon"]==j+lon.min().item())]["cover"]
        if len(clist)==0:
            pass
        else:
            cover_max[i,j] = clist.max()
            cover_min[i,j] = clist.min()
            cover_mean[i,j] = clist.mean()
            cover_size[i,j] = len(clist)
            
            
f = lambda x: round(x,2)

save_hm(pd.DataFrame(cover_max,
                         index=pd.Series(np.arange(lat.min().item(),lat.max().item())*(1/size)).map(f),
                         columns=pd.Series(np.arange(lon.min().item(),lon.max().item())*(1/size)).map(f))[::-1]
       ,"cover_max")

save_hm(pd.DataFrame(cover_min,
                         index=pd.Series(np.arange(lat.min().item(),lat.max().item())*(1/size)).map(f),
                         columns=pd.Series(np.arange(lon.min().item(),lon.max().item())*(1/size)).map(f))[::-1]
       ,"cover_min")
       

save_hm(pd.DataFrame(cover_mean,
                         index=pd.Series(np.arange(lat.min().item(),lat.max().item())*(1/size)).map(f),
                         columns=pd.Series(np.arange(lon.min().item(),lon.max().item())*(1/size)).map(f))[::-1]
       ,"cover_mean")

save_hm(pd.DataFrame(cover_size,
                         index=pd.Series(np.arange(lat.min().item(),lat.max().item())*(1/size)).map(f),
                         columns=pd.Series(np.arange(lon.min().item(),lon.max().item())*(1/size)).map(f))[::-1]
       ,"cover_size")
