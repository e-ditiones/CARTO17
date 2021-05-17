import folium, os
import pandas as pd
from folium.plugins import MarkerCluster

#get data
data = pd.read_csv(os.path.join('listPlaces','listPlaces_classified.tsv'), delimiter='\t')
data['prediction'] = data['prediction'].replace([0],'red')
data['prediction'] = data['prediction'].replace([1],'green')

#same data but counting duplicated rows
data_subset = data[["prediction", "long", "lat","placeName"]]
data2=data_subset.groupby(data_subset.columns.tolist(),as_index=False).size()

#center for the map (coordiantes of Rome)
center = (41.902782, 12.496366)

# create empty map zoomed in on center
map = folium.Map(location=center,
									zoom_start=4,
                  tiles='Stamen Watercolor')

# add a marker for every record in the filtered data, use a clustered view
for each in data.iterrows():
    folium.Circle(location = [each[1]['long'],each[1]['lat']],
    								radius=25000,
    								color=each[1]['prediction'],
    								fill_color=each[1]['prediction'],
    								fill=True,
    								popup=each[1]['placeName']
    							).add_to(map)

map.save("map.html")

# create empty map zoomed in on center with circles proportional to occurrences
map = folium.Map(location=center,
									zoom_start=4,
                  tiles='Stamen Watercolor')

# add a marker for every record in the filtered data, use a clustered view
for each in data2.iterrows():
    folium.Circle(location = [each[1]['long'],each[1]['lat']],
    								radius=each[1]['size']*100,
    								color=each[1]['prediction'],
    								fill_color=each[1]['prediction'],
    								fill=True,
    								popup=each[1]['placeName']
    							).add_to(map)

map.save("map2.html")
