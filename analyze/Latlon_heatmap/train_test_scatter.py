#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
	

df_train = pd.read_csv("../../official_data/train_data.csv")[["lat","lon"]]
df_test = pd.read_csv("../../official_data/test_data.csv")[["lat","lon"]]
plt.scatter(df_train["lon"],df_train["lat"],s=3,c="Blue")
plt.scatter(df_test["lon"],df_test["lat"],s=3,c="Red")
plt.title("train_test_scatter")
plt.savefig("train_test_scatter")
