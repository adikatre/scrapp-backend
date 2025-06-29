import pandas as pd
import sqlite3

conn = sqlite3.connect("data/FPA_FOD.sqlite")
df = pd.read_sql_query("SELECT NWCG_REPORTING_UNIT_NAME,FIRE_NAME,STAT_CAUSE_DESCR,LATITUDE,LONGITUDE,STATE,DISCOVERY_DATE FROM 'FIRES'", conn)

#replace these values later; this is a proof of concept for Los angeles
lat = 37.2740
lon = -107.8792 #west is -ve

#search area radius (degrees)
radius = 0.5

nearby_fires = df[(df['LATITUDE'].between(lat - radius, lat + radius)) & (df['LONGITUDE'].between(lon - radius, lon + radius))]

print(f"Number of fires near ({lat}, {lon}): {len(nearby_fires)}")
print("\nTop causes:")
print(nearby_fires['STAT_CAUSE_DESCR'].value_counts())

