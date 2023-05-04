#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from folium import Map, Marker, Icon
from folium.plugins import MarkerCluster
from pathlib import Path
import matplotlib.cm

data_path = Path('')

original_train_data = pd.read_csv('../../official_data/train_data.csv', usecols=['lat', 'lon', 'cover', 'year'])
original_test_data = pd.read_csv('../../official_data/test_data.csv', usecols=['lat', 'lon', 'year'])

map_train = Map(
    location=[25.5, 127],
    zoom_start=8,
    )

marker_cluster = MarkerCluster().add_to(map_train)
train_lat = original_train_data['lat'].to_numpy()
train_lon = original_train_data['lon'].to_numpy()
train_cover = original_train_data['cover'].to_numpy()

for lat, lon, cover in zip(train_lat, train_lon, train_cover):
    Marker(
        location=[lat, lon],
        popup=f"({lat}, {lon})",
        tooltip=cover
        ).add_to(marker_cluster)

map_train.save("map_ocean_train.html")



map_test = Map(
    location=[25.5, 127],
    zoom_start=8,
    )

marker_cluster = MarkerCluster().add_to(map_test)

test_lat = original_test_data['lat'].to_numpy()
test_lon = original_test_data['lon'].to_numpy()

for lat, lon in zip(test_lat, test_lon):
    Marker(
        location=[lat, lon],
        popup=f"({lat}, {lon})",
        icon=Icon('red', icon='star'),
        ).add_to(marker_cluster)

map_test.save("map_ocean_test.html")
